"""
prepare_data.py
Download unsloth/Radiology_mini, save images to disk,
and build HuggingFace datasets in chat-template format for TRL GRPOTrainer.
"""

import os
from datasets import load_dataset

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "radiology_images")
SYSTEM_PROMPT = (
    "You are a radiology assistant. Describe the findings in the given "
    "medical image accurately and concisely."
)
USER_QUESTION = "Describe what you observe in this radiology image."


def save_image(image, image_id: str, split: str) -> str:
    """Save a PIL image to disk and return its file path."""
    split_dir = os.path.join(IMAGE_DIR, split)
    os.makedirs(split_dir, exist_ok=True)
    path = os.path.join(split_dir, f"{image_id}.png")
    if not os.path.exists(path):
        image.save(path)
    return path


def format_example(example, split: str):
    """Convert a raw dataset row into GRPOTrainer-compatible format."""
    image_path = save_image(example["image"], example["image_id"], split)

    # Build multimodal chat prompt
    example["prompt"] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": USER_QUESTION},
            ],
        },
    ]

    # Ground truth caption for reward scoring
    example["ground_truth"] = example["caption"]
    return example


def main():
    print("Loading unsloth/Radiology_mini ...")
    ds_train = load_dataset("unsloth/Radiology_mini", split="train")
    ds_test = load_dataset("unsloth/Radiology_mini", split="test")

    print(f"Train: {len(ds_train)} samples, Test: {len(ds_test)} samples")
    print(f"Columns: {ds_train.column_names}")

    print("Formatting train split ...")
    ds_train = ds_train.map(
        lambda ex: format_example(ex, "train"),
        desc="Formatting train",
    )

    print("Formatting test split ...")
    ds_test = ds_test.map(
        lambda ex: format_example(ex, "test"),
        desc="Formatting test",
    )

    # Keep only the columns needed by the trainer
    keep_cols = ["prompt", "ground_truth"]
    ds_train = ds_train.remove_columns(
        [c for c in ds_train.column_names if c not in keep_cols]
    )
    ds_test = ds_test.remove_columns(
        [c for c in ds_test.column_names if c not in keep_cols]
    )

    print(f"\nFinal train schema: {ds_train.column_names}, len={len(ds_train)}")
    print(f"Final test schema:  {ds_test.column_names}, len={len(ds_test)}")
    print(f"Sample prompt:\n{ds_train[0]['prompt']}")
    print(f"Sample ground_truth:\n{ds_train[0]['ground_truth'][:200]}")

    # Save processed datasets
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    ds_train.save_to_disk(os.path.join(out_dir, "train"))
    ds_test.save_to_disk(os.path.join(out_dir, "test"))
    print(f"\nSaved processed datasets to {out_dir}/")


if __name__ == "__main__":
    main()
