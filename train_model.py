from pathlib import Path
import argparse
import csv
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from monai.data import Dataset, CacheDataset, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)
from monai.utils import set_determinism


def parse_roi_size(text):
    values = tuple(int(x.strip()) for x in text.split(","))

    if len(values) != 3:
        raise ValueError("roi_size must have three values")

    return values


def get_train_transforms(roi_size):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),

            Orientationd(keys=["image", "label"], axcodes="RAS"),

            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),

            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
            ),

            SpatialPadd(
                keys=["image", "label"],
                spatial_size=roi_size,
            ),

            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),

            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.10),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.10),

            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),

            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),

            EnsureTyped(keys=["image", "label"]),
        ]
    )


def get_val_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),

            Orientationd(keys=["image", "label"], axcodes="RAS"),

            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),

            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
            ),

            EnsureTyped(keys=["image", "label"]),
        ]
    )


def validate(model, val_loader, loss_fn, args, device):
    model.eval()

    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(
        include_background=False,
        percentile=95,
        reduction="mean",
    )

    losses = []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if i == 0:
                print("val image shape:", images.shape)
                print("val label shape:", labels.shape)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                outputs = sliding_window_inference(
                    inputs=images,
                    roi_size=args.roi_size,
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=0.5,
                )

                loss = loss_fn(outputs, labels)

            losses.append(loss.item())

            outputs_list = [post_pred(x) for x in decollate_batch(outputs)]
            labels_list = [post_label(x) for x in decollate_batch(labels)]

            dice_metric(y_pred=outputs_list, y=labels_list)
            iou_metric(y_pred=outputs_list, y=labels_list)

            try:
                hd95_metric(y_pred=outputs_list, y=labels_list)
            except Exception:
                pass

    try:
        val_dice = float(dice_metric.aggregate().item())
    except Exception:
        val_dice = None

    try:
        val_iou = float(iou_metric.aggregate().item())
    except Exception:
        val_iou = None

    try:
        val_hd95 = float(hd95_metric.aggregate().item())
    except Exception:
        val_hd95 = None

    dice_metric.reset()
    iou_metric.reset()
    hd95_metric.reset()

    val_loss = float(np.mean(losses)) if len(losses) > 0 else None

    return val_loss, val_dice, val_iou, val_hd95


def save_prediction_preview(model, val_loader, args, device, output_path):
    model.eval()

    output_path.parent.mkdir(exist_ok=True)

    batch = next(iter(val_loader))

    image = batch["image"].to(device)
    label = batch["label"].to(device)

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=image,
            roi_size=args.roi_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model,
            overlap=0.5,
        )

    pred = torch.argmax(output, dim=1, keepdim=True)

    image_np = image[0, 0].detach().cpu().numpy()
    label_np = label[0, 0].detach().cpu().numpy()
    pred_np = pred[0, 0].detach().cpu().numpy()

    if label_np.sum() > 0:
        slice_id = int(np.argmax(label_np.sum(axis=(0, 1))))
    else:
        slice_id = image_np.shape[-1] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image_np[:, :, slice_id], cmap="gray")
    axes[0].set_title("CT slice")
    axes[0].axis("off")

    axes[1].imshow(image_np[:, :, slice_id], cmap="gray")
    axes[1].imshow(label_np[:, :, slice_id], alpha=0.4)
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(image_np[:, :, slice_id], cmap="gray")
    axes[2].imshow(pred_np[:, :, slice_id], alpha=0.4)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("saved prediction preview:", output_path)


def train(args):
    project = Path(__file__).resolve().parent

    results_dir = project / "results"
    checkpoint_dir = results_dir / "checkpoints"
    figures_dir = results_dir / "figures"

    split_file = results_dir / "data_splits.json"
    metrics_file = results_dir / "metrics.csv"

    results_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_determinism(seed=args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("quick test:", args.quick_test)
    print("roi size:", args.roi_size)
    print("amp:", args.amp)

    # load data split
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            "Run data-preparation.py first."
        )

    with open(split_file, "r") as f:
        split_data = json.load(f)

    train_data = split_data["train"]
    val_data = split_data["val"]

    if args.quick_test:
        train_data = train_data[:4]
        val_data = val_data[:2]

    print("training cases:", len(train_data))
    print("validation cases:", len(val_data))

    if len(train_data) > 0:
        print("first train file:", train_data[0])

    if len(val_data) > 0:
        print("first val file:", val_data[0])

    # datasets
    if args.cache_rate > 0:
        train_ds = CacheDataset(
            data=train_data,
            transform=get_train_transforms(args.roi_size),
            cache_rate=args.cache_rate,
            num_workers=0,
        )

        val_ds = CacheDataset(
            data=val_data,
            transform=get_val_transforms(),
            cache_rate=args.cache_rate,
            num_workers=0,
        )
    else:
        train_ds = Dataset(
            data=train_data,
            transform=get_train_transforms(args.roi_size),
        )

        val_ds = Dataset(
            data=val_data,
            transform=get_val_transforms(),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("train batches:", len(train_loader))
    print("val batches:", len(val_loader))

    # model
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=args.epochs,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=args.amp and device.type == "cuda"
    )

    best_dice = -1.0
    history = []

    best_model_path = checkpoint_dir / "best_model.pth"
    last_model_path = checkpoint_dir / "last_model.pth"

    # training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        print("-" * 50)
        print("epoch", epoch, "/", args.epochs)

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if epoch == 1 and step == 1:
                print("train image shape:", images.shape)
                print("train label shape:", labels.shape)
                print("image min/max:", images.min().item(), images.max().item())
                print("label values:", torch.unique(labels).detach().cpu().numpy())

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            losses.append(loss.item())

            if step % args.log_every == 0:
                print(
                    "epoch:",
                    epoch,
                    "step:",
                    step,
                    "/",
                    len(train_loader),
                    "loss:",
                    round(loss.item(), 4),
                )

        train_loss = float(np.mean(losses))
        scheduler.step()

        val_loss = None
        val_dice = None
        val_iou = None
        val_hd95 = None

        # validation part
        if epoch % args.val_interval == 0:
            val_loss, val_dice, val_iou, val_hd95 = validate(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                args=args,
                device=device,
            )

            print("epoch:", epoch)
            print("train loss:", train_loss)
            print("val loss:", val_loss)
            print("dice:", val_dice)
            print("iou:", val_iou)
            print("hd95:", val_hd95)

            # save best model
            if val_dice is not None and val_dice > best_dice:
                best_dice = val_dice

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_dice": best_dice,
                        "args": vars(args),
                    },
                    best_model_path,
                )

                print("saved best model:", best_model_path)

                save_prediction_preview(
                    model=model,
                    val_loader=val_loader,
                    args=args,
                    device=device,
                    output_path=figures_dir / "best_prediction_preview.png",
                )

        current_lr = opt.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice": val_dice,
                "val_iou": val_iou,
                "val_hd95": val_hd95,
                "learning_rate": current_lr,
            }
        )

        # save metrics
        with open(metrics_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "val_dice",
                    "val_iou",
                    "val_hd95",
                    "learning_rate",
                ],
            )

            writer.writeheader()

            for row in history:
                writer.writerow(row)

        # save last model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_dice": best_dice,
                "args": vars(args),
            },
            last_model_path,
        )

    print("training finished")
    print("best validation dice:", best_dice)
    print("metrics saved to:", metrics_file)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--roi-size", type=parse_roi_size, default=(96, 96, 96))
    parser.add_argument("--sw-batch-size", type=int, default=1)

    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-rate", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)

    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--amp", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.quick_test:
        args.epochs = min(args.epochs, 2)

    train(args)