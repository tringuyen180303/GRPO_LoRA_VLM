"""
Prepare radiology data for verl GRPO / PPO.

Outputs:
  data/train.parquet
  data/test.parquet
"""

import os
from io import BytesIO

from datasets import load_dataset

BASE_DIR = os.path.dirname(__file__)
IMAGE_DIR = os.path.join(BASE_DIR, "radiology_images")
OUT_DIR = os.path.join(BASE_DIR, "data")

DATASET_NAME = "unsloth/Radiology_mini"

SYSTEM_PROMPT = (
    "You are a radiology assistant. Describe the findings in the given "
    "medical image accurately and concisely."
)
USER_QUESTION = "Describe what you observe in this radiology image."


def image_to_bytes(image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def save_image(image, image_id: str, split: str) -> str:
    split_dir = os.path.join(IMAGE_DIR, split)
    os.makedirs(split_dir, exist_ok=True)
    path = os.path.join(split_dir, f"{image_id}.png")
    if not os.path.exists(path):
        image.save(path)
    return path


def format_example(example, idx: int, split: str):
    image = example["image"]
    image_id = str(example.get("image_id", idx))

    image_path = save_image(image, image_id, split)
    image_bytes = image_to_bytes(image)

    return {
        "data_source": DATASET_NAME,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<image>\n{USER_QUESTION}",
            },
        ],
        # verl default loader expects the multimodal content in this column
        # and _build_messages() can consume dicts that contain "bytes".
        "images": [
            {
                "bytes": image_bytes,
                "path": image_path,
            }
        ],
        "ability": "radiology",
        "reward_model": {
            "style": "rule",
            "ground_truth": example["caption"],
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "image_id": image_id,
        },
    }


def main():
    print(f"Loading {DATASET_NAME} ...")
    ds = load_dataset(DATASET_NAME)

    train_ds = ds["train"].map(
        lambda ex, idx: format_example(ex, idx, "train"),
        with_indices=True,
        remove_columns=ds["train"].column_names,
        desc="Formatting train",
    )

    test_ds = ds["test"].map(
        lambda ex, idx: format_example(ex, idx, "test"),
        with_indices=True,
        remove_columns=ds["test"].column_names,
        desc="Formatting test",
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    train_path = os.path.join(OUT_DIR, "train.parquet")
    test_path = os.path.join(OUT_DIR, "test.parquet")

    train_ds.to_parquet(train_path)
    test_ds.to_parquet(test_path)

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")
    print("Columns:", train_ds.column_names)
    print("Sample:", train_ds[0])


if __name__ == "__main__":
    main()
