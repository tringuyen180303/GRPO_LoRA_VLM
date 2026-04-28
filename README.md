# GRPO + LoRA Fine-tuning of Vision Language Models for Radiology Captioning

Fine-tune a Vision Language Model (VLM) to generate accurate radiology image captions using Group Relative Policy Optimization (GRPO) and Low-Rank Adaptation (LoRA). Reward is computed via BioBERT cosine similarity against ground-truth captions — no human labels required at training time.

---

## Overview

| Component | Description |
|---|---|
| **Base model** | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| **Dataset** | `unsloth/Radiology_mini` (HuggingFace) |
| **RL algorithm** | GRPO — samples G completions per prompt, normalizes rewards into advantages |
| **Adapter** | LoRA on attention + MLP layers of the language model (vision encoder frozen) |
| **Reward** | BioBERT cosine similarity (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`) |

Two training backends are provided:

- **`train_grpo_raw.py`** — pure PyTorch, no TRL or PEFT dependencies (recommended)
- **`main.py`** — TRL `GRPOTrainer` + PEFT on `Qwen/Qwen2.5-VL-3B-Instruct` (requires GPU with sufficient VRAM)

---

## Project Structure

```
GRPO_LoRA_VLM/
├── prepare_data.py          # Download & format Radiology_mini → data/
├── train_grpo_raw.py        # Pure PyTorch GRPO + LoRA training (SmolVLM-256M)
├── main.py                  # TRL GRPOTrainer + PEFT training (Qwen2.5-VL-3B)
├── lora_utils.py            # Custom LoRA injection, save, and load utilities
├── reward_biobert.py        # BioBERT cosine similarity reward function
├── evaluation/
│   └── evaluating.py        # Evaluate a checkpoint on the test split
├── xray_grpo_lora_smolvlm/  # Saved checkpoints from train_grpo_raw.py
│   ├── lora_final.pt
│   └── lora_step*.pt
└── data/                    # Processed dataset (created by prepare_data.py)
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers datasets pillow sentencepiece accelerate trl peft
```

> **HPC / older PyTorch note:** If your environment uses PyTorch < 2.6, the `torch.load` CVE patch in newer `transformers` will block `.pt` loading. The code works around this by using `use_safetensors=True` for BioBERT. Ensure your HuggingFace cache has the `model.safetensors` variant downloaded.

---

## Quick Start

### 1. Prepare the dataset

Downloads `unsloth/Radiology_mini`, saves images to disk, and builds HuggingFace datasets in chat-template format.

```bash
python prepare_data.py
```

Output: `data/train/` and `data/test/`

---

### 2. Train (pure PyTorch — recommended)

```bash
python train_grpo_raw.py
```

Key hyperparameters (edit `TrainConfig` in `train_grpo_raw.py`):

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `HuggingFaceTB/SmolVLM-256M-Instruct` | Base VLM |
| `lora_r` | `32` | LoRA rank |
| `lora_alpha` | `32.0` | LoRA scaling |
| `num_generations` | `4` | Completions sampled per prompt (GRPO group size) |
| `beta` | `0.001` | KL penalty coefficient |
| `num_epochs` | `15` | Training epochs |
| `lr` | `1e-5` | Learning rate |
| `max_new_tokens` | `128` | Max caption length |

Checkpoints are saved to `xray_grpo_lora_smolvlm/`:
- `lora_step{N}.pt` every 50 steps
- `lora_final.pt` at the end of training

---

### 3. Evaluate

```bash
# Fine-tuned model (lora_final.pt)
python evaluation/evaluating.py

# Base model without LoRA (baseline)
python evaluation/evaluating.py --base
```

Results are written to `evaluation/results_lora.txt` or `evaluation/results_base.txt` with per-sample BioBERT similarity scores and an average at the end. Progress is printed to the terminal.

---

## How GRPO Works Here

For each training prompt (image + instruction):

1. **Sample** G=4 captions from the current policy model.
2. **Score** each caption with BioBERT cosine similarity against the ground-truth caption.
3. **Normalize** rewards within the group: `A_i = (r_i − μ) / σ`.
4. **Update** the policy to increase probability of high-advantage completions while penalizing KL divergence from the frozen reference model:

```
L = −mean(A_i · log π_θ(y_i | x)) + β · mean(log π_θ − log π_ref)
```

---

## LoRA Configuration

LoRA is injected into the language model's attention and MLP projections only — the vision encoder stays fully frozen throughout training.

**Target layers:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

**Skipped:** `model.vision_model.*`

---

## Reward Model

`reward_biobert.py` encodes both the generated caption and the ground-truth caption with BioBERT, then returns their cosine similarity clamped to `[0, 1]` as the reward signal.

```python
from reward_biobert import biobert_reward_fn

scores = biobert_reward_fn(
    ["Bilateral pleural effusion with cardiomegaly."],
    ground_truth=["Large bilateral pleural effusions and enlarged cardiac silhouette."]
)
# → [0.98]
```

The BioBERT model loads lazily on first call and is cached as a singleton for the duration of training.
