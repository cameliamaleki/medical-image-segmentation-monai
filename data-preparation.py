from pathlib import Path
import csv
import json
import random

import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split


project = Path(__file__).resolve().parent

data_dir = project / "Task09_Spleen"
dataset_json = data_dir / "dataset.json"

results = project / "results"
figures = results / "figures"

split_file = results / "data_splits.json"
qc_file = results / "dataset_qc.csv"
summary_file = results / "dataset_summary.txt"

seed = 42
val_size = 0.2


def check_one_case(case):
    image_path = Path(case["image"])
    label_path = Path(case["label"])

    row = {
        "case": case["case"],
        "image_path": str(image_path),
        "label_path": str(label_path),
        "status": "ok",
    }

    try:
        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))

        image = image_nii.get_fdata(dtype=np.float32)
        label = label_nii.get_fdata(dtype=np.float32)

        image_shape = image.shape
        label_shape = label.shape

        image_spacing = tuple(float(x) for x in image_nii.header.get_zooms()[:3])
        label_spacing = tuple(float(x) for x in label_nii.header.get_zooms()[:3])

        label_mask = label > 0
        label_values = np.unique(label)

        voxel_volume_mm3 = np.prod(label_spacing)
        spleen_volume_ml = label_mask.sum() * voxel_volume_mm3 / 1000.0

        p1, p50, p99 = np.percentile(image, [1, 50, 99])

        row.update(
            {
                "image_shape": str(image_shape),
                "label_shape": str(label_shape),
                "image_spacing": str(image_spacing),
                "label_spacing": str(label_spacing),
                "shape_match": image_shape == label_shape,
                "spacing_match": image_spacing == label_spacing,
                "image_min": float(np.min(image)),
                "image_p1": float(p1),
                "image_median": float(p50),
                "image_p99": float(p99),
                "image_max": float(np.max(image)),
                "label_values": str(label_values.tolist()),
                "label_voxels": int(label_mask.sum()),
                "spleen_volume_ml": float(spleen_volume_ml),
            }
        )

        if image_shape != label_shape:
            row["status"] = "shape_mismatch"

        elif label_mask.sum() == 0:
            row["status"] = "empty_label"

    except Exception as e:
        row["status"] = "error"
        row["error"] = str(e)

    return row


def save_qc_summary(rows):
    ok_rows = [row for row in rows if row.get("status") == "ok"]

    volumes = [
        float(row["spleen_volume_ml"])
        for row in ok_rows
        if "spleen_volume_ml" in row
    ]

    status_counts = {}

    for row in rows:
        status = row.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    with open(summary_file, "w") as f:
        f.write("Dataset summary\n")
        f.write("================\n\n")

        f.write(f"Total cases: {len(rows)}\n")
        f.write(f"Valid cases: {len(ok_rows)}\n")
        f.write(f"Problem cases: {len(rows) - len(ok_rows)}\n\n")

        f.write("Status counts:\n")

        for status, count in status_counts.items():
            f.write(f"- {status}: {count}\n")

        if len(volumes) > 0:
            f.write("\nSpleen volume, ml:\n")
            f.write(f"- mean: {np.mean(volumes):.2f}\n")
            f.write(f"- median: {np.median(volumes):.2f}\n")
            f.write(f"- min: {np.min(volumes):.2f}\n")
            f.write(f"- max: {np.max(volumes):.2f}\n")

    print("saved summary file:", summary_file)


def main():
    results.mkdir(exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder was not found:\n{data_dir}")

    if not dataset_json.exists():
        raise FileNotFoundError(f"dataset.json was not found:\n{dataset_json}")

    print("loading dataset:", dataset_json)

    with open(dataset_json, "r") as f:
        dataset_info = json.load(f)

    training_cases = dataset_info["training"]
    data_list = []

    for case in training_cases:
        image_path = data_dir / case["image"].replace("./", "")
        label_path = data_dir / case["label"].replace("./", "")

        name = image_path.name

        if name.endswith(".nii.gz"):
            case_name = name.replace(".nii.gz", "")
        else:
            case_name = image_path.stem

        if image_path.exists() and label_path.exists():
            data_list.append(
                {
                    "case": case_name,
                    "image": str(image_path),
                    "label": str(label_path),
                }
            )
        else:
            print("missing file:")
            print(image_path)
            print(label_path)

    if len(data_list) == 0:
        raise ValueError("No valid image-label pairs were found.")

    print("total valid files:", len(data_list))
    print("running dataset quality check...")

    rows = []

    for i, case in enumerate(data_list, start=1):
        row = check_one_case(case)
        rows.append(row)

        print(
            "checked",
            i,
            "/",
            len(data_list),
            "|",
            row["case"],
            "|",
            row["status"],
        )

    # save qc table
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))

    with open(qc_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print("saved qc file:", qc_file)

    save_qc_summary(rows)

    random.seed(seed)

    train_files, val_files = train_test_split(
        data_list,
        test_size=val_size,
        random_state=seed,
        shuffle=True,
    )

    print("training cases:", len(train_files))
    print("validation cases:", len(val_files))

    # save split for MONAI
    split_data = {
        "train": train_files,
        "val": val_files,
    }

    with open(split_file, "w") as f:
        json.dump(split_data, f, indent=4)

    print("saved split file:", split_file)


if __name__ == "__main__":
    main()