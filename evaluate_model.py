from pathlib import Path
import argparse
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
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
    ScaleIntensityRanged,
    Spacingd,
)


project = Path(__file__).resolve().parent

split_file = project / "results" / "data_splits.json"
checkpoint_path = project / "results" / "checkpoints" / "best_model.pth"

prediction_dir = project / "results" / "predictions"
figure_dir = project / "results" / "figures"
case_metrics_file = project / "results" / "case_metrics.csv"

prediction_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

roi_size = (96, 96, 96)


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


def create_loader(limit=None):
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            "Run data-preparation.py first."
        )

    with open(split_file, "r") as f:
        split_data = json.load(f)

    val_files = split_data["val"]

    if limit is not None:
        val_files = val_files[:limit]

    val_ds = Dataset(
        data=val_files,
        transform=get_val_transforms(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    print("validation cases loaded:", len(val_files))

    return val_loader, val_files


def create_model():
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )

    return model


def load_model(device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            "Train the model first using train_model.py."
        )

    model = create_model().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print("loaded model from:", checkpoint_path)

    return model


def calculate_dice(prediction, label):
    post_pred = AsDiscrete(to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
    )

    pred_list = [post_pred(x) for x in decollate_batch(prediction)]
    label_list = [post_label(x) for x in decollate_batch(label)]

    dice_metric(y_pred=pred_list, y=label_list)

    dice_value = dice_metric.aggregate().item()
    dice_metric.reset()

    return dice_value


def save_case_figure(image, label, prediction, case_name, dice_value):
    image_np = image[0, 0].detach().cpu().numpy()
    label_np = label[0, 0].detach().cpu().numpy()
    pred_np = prediction[0, 0].detach().cpu().numpy()

    if label_np.sum() > 0:
        slice_id = int(np.argmax(label_np.sum(axis=(0, 1))))
    else:
        slice_id = label_np.shape[-1] // 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(image_np[:, :, slice_id], cmap="gray")
    axes[0].set_title("CT slice")
    axes[0].axis("off")

    axes[1].imshow(image_np[:, :, slice_id], cmap="gray")
    axes[1].imshow(label_np[:, :, slice_id], alpha=0.45)
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(image_np[:, :, slice_id], cmap="gray")
    axes[2].imshow(pred_np[:, :, slice_id], alpha=0.45)
    axes[2].set_title(f"Prediction | Dice {dice_value:.3f}")
    axes[2].axis("off")

    plt.tight_layout()

    save_path = figure_dir / f"{case_name}_prediction.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    print("saved figure:", save_path)


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = load_model(device)
    val_loader, val_files = create_loader(limit=args.limit)

    rows = []

    with torch.no_grad():
        for case_id, batch in enumerate(val_loader):
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            output = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

            prediction = torch.argmax(output, dim=1, keepdim=True)

            dice_value = calculate_dice(
                prediction=prediction,
                label=label,
            )

            image_path = Path(val_files[case_id]["image"])
            case_name = image_path.stem.replace(".nii", "")

            print("case:", case_name, "| dice:", round(dice_value, 4))

            rows.append(
                {
                    "case": case_name,
                    "dice": dice_value,
                }
            )

            save_case_figure(
                image=image,
                label=label,
                prediction=prediction,
                case_name=case_name,
                dice_value=dice_value,
            )

    # save case metrics
    with open(case_metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "dice"],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print("saved case metrics:", case_metrics_file)

    if len(rows) > 0:
        mean_dice = np.mean([row["dice"] for row in rows])
        print("mean dice:", round(mean_dice, 4))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of validation cases to run. Leave empty to run all.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    run_inference(args)