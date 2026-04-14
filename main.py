"""
main.py
GRPO + LoRA fine-tuning of a VLM on Radiology_mini with BioBERT reward.

Usage:
    1. python prepare_data.py        # download & format dataset (one-time)
    2. python main.py                # run GRPO training
"""

import os
from datasets import load_from_disk, load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor



from reward_biobert import biobert_reward_fn

# ─── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = "./xray_grpo_lora"

SYSTEM_PROMPT = (
    "You are a radiology assistant. Describe the findings in the given "
    "medical image accurately and concisely."
)
USER_QUESTION = "Describe what you observe in this radiology image."


# ─── Data loading ───────────────────────────────────────────────────────────────
def load_prepared_data():
    """Load pre-processed data from prepare_data.py output."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    train_path = os.path.join(data_dir, "train")
    test_path = os.path.join(data_dir, "test")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading pre-processed data from disk ...")
        return load_from_disk(train_path), load_from_disk(test_path)
    else:
        print("Pre-processed data not found. Building on-the-fly ...")
        return load_and_format_data()


def load_and_format_data():
    """Fallback: load and format data directly (no disk save)."""
    from prepare_data import format_example, save_image  # noqa: F401

    ds_train = load_dataset("unsloth/Radiology_mini", split="train")
    ds_test = load_dataset("unsloth/Radiology_mini", split="test")

    ds_train = ds_train.map(lambda ex: format_example(ex, "train"))
    ds_test = ds_test.map(lambda ex: format_example(ex, "test"))

    keep_cols = ["prompt", "ground_truth"]
    ds_train = ds_train.remove_columns(
        [c for c in ds_train.column_names if c not in keep_cols]
    )
    ds_test = ds_test.remove_columns(
        [c for c in ds_test.column_names if c not in keep_cols]
    )
    return ds_train, ds_test


# ─── Main ────────────────────────────────────────────────────────────────────────
def main():
    # Load processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # LoRA config — exclude vision encoder to keep it frozen
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        # Freeze the vision encoder entirely — only finetune language model
        modules_to_save=None,
    )

    # Load datasets
    train_ds, val_ds = load_prepared_data()
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # GRPO training config
    grpo_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,              # Group size for GRPO
 #       max_prompt_length=1024,
 #       max_completion_length=128,      # Captions are short
        num_train_epochs=3,
        bf16=True,
        logging_steps=5,
        save_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        warmup_ratio=0.1,
        report_to="none",
        gradient_checkpointing=True,
        # GRPO-specific: KL penalty
        beta=0.001,                     # KL coefficient
    )

    # Initialize GRPO trainer with BioBERT reward
    trainer = GRPOTrainer(
        model=MODEL_NAME,
        args=grpo_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        reward_funcs=biobert_reward_fn,
        peft_config=lora_config,
    )

    print("\n" + "=" * 60)
    print("Starting GRPO + LoRA training with BioBERT reward")
    print(f"  Model:       {MODEL_NAME}")
    print(f"  LoRA rank:   {lora_config.r}")
    print(f"  Group size:  {grpo_args.num_generations}")
    print(f"  LR:          {grpo_args.learning_rate}")
    print(f"  Output:      {OUTPUT_DIR}")
    print("=" * 60 + "\n")

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print(f"\nTraining complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()