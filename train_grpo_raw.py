"""
train_grpo_raw.py
GRPO + LoRA fine-tuning of SmolVLM-256M on Radiology_mini.
Pure PyTorch — no TRL, no PEFT.

Usage
-----
    python prepare_data.py       # one-time: downloads & formats dataset
    python train_grpo_raw.py     # trains the model

What replaces what
------------------
    peft.LoraConfig / get_peft_model  →  lora_utils.inject_lora
    trl.GRPOTrainer / GRPOConfig      →  grpo_step() + train() below
"""

import copy
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_from_disk
from PIL import Image

from lora_utils import (
    inject_lora,
    get_trainable_params,
    count_params,
    save_lora_weights,
)
from reward_biobert import biobert_reward_fn


# ─── Config ──────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Model
    model_name: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    output_dir: str = "./xray_grpo_lora_smolvlm"

    # LoRA — only language-model layers; vision encoder is always frozen
    lora_r: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",       # MLP
    )

    # GRPO
    num_generations: int = 4   # G completions sampled per prompt
    beta: float = 0.001        # KL-penalty coefficient

    # Training loop
    num_epochs: int = 15
    lr: float = 1e-5
    warmup_ratio: float = 0.1
    grad_accum_steps: int = 4  # effective mini-batch = this many prompts
    max_grad_norm: float = 1.0

    # Generation
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_p: float = 0.9

    # Misc
    bf16: bool = True
    log_steps: int = 5
    save_steps: int = 50
    seed: int = 42
    data_dir: str = "./data"


# ─── Data helpers ─────────────────────────────────────────────────────────────────

def load_data(cfg: TrainConfig):
    train_ds = load_from_disk(os.path.join(cfg.data_dir, "train"))
    val_ds   = load_from_disk(os.path.join(cfg.data_dir, "test"))
    return train_ds, val_ds


def _extract_images(messages: list) -> list[Image.Image]:
    """Walk message dicts and return PIL images in content order."""
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image":
                continue
            img = part["image"]
            if isinstance(img, str):          # path on disk
                img = Image.open(img).convert("RGB")
            images.append(img)
    return images


def build_prompt_inputs(
    messages: list,
    processor: AutoProcessor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Convert a list-of-message-dicts (system + user with image) into a dict of
    tensors ready for Qwen2.5-VL — with the generation prompt appended.
    """
    images = _extract_images(messages)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=images or None,
        return_tensors="pt",
        padding=True,
    )
    return {k: v.to(device) for k, v in inputs.items()}


# ─── Log-probability computation ──────────────────────────────────────────────────

def compute_seq_log_prob(
    model: torch.nn.Module,
    prompt_inputs: dict[str, torch.Tensor],
    completion_ids: torch.Tensor,   # shape (1, T)
) -> torch.Tensor:
    """
    Return  Σ_t  log π(token_t | context)  summed over the T completion tokens.

    Runs one forward pass on the concatenated [prompt || completion] sequence.
    Gradients flow when the model is in train mode; wrap in torch.no_grad()
    externally when scoring the reference model.

    Index arithmetic
    ----------------
    logits[:, j, :]  =  distribution over token at position j+1.
    Completion tokens live at positions  [L, L+T-1]  (0-indexed).
    Their predicting logits are at       [L-1, L+T-2].
    → comp_logits = logits[:, L-1 : L+T-1, :]
    """
    L = prompt_inputs["input_ids"].shape[1]   # prompt length
    T = completion_ids.shape[1]               # completion length

    full_ids  = torch.cat([prompt_inputs["input_ids"],  completion_ids],          dim=1)
    full_mask = torch.cat([prompt_inputs["attention_mask"], torch.ones_like(completion_ids)], dim=1)

    kwargs: dict = {"input_ids": full_ids, "attention_mask": full_mask}
    if "pixel_values" in prompt_inputs:
        kwargs["pixel_values"] = prompt_inputs["pixel_values"]
    if "pixel_attention_mask" in prompt_inputs:
        kwargs["pixel_attention_mask"] = prompt_inputs["pixel_attention_mask"]
    if "image_grid_thw" in prompt_inputs:
        kwargs["image_grid_thw"] = prompt_inputs["image_grid_thw"]

    outputs = model(**kwargs)                              # logits: (1, L+T, V)
    comp_logits = outputs.logits[:, L - 1 : L + T - 1, :]  # (1, T, V)

    log_probs   = F.log_softmax(comp_logits, dim=-1)       # (1, T, V)
    token_lp    = log_probs.gather(2, completion_ids.unsqueeze(-1)).squeeze(-1)  # (1, T)
    return token_lp.sum(dim=-1)                            # (1,)


# ─── GRPO step ────────────────────────────────────────────────────────────────────

def grpo_step(
    policy: torch.nn.Module,
    ref: torch.nn.Module,
    prompt_inputs: dict[str, torch.Tensor],
    ground_truth: str,
    processor: AutoProcessor,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """
    One GRPO optimisation step for a single prompt.

    Algorithm
    ---------
    1. Sample G completions from the current policy (no grad).
    2. Score each completion with the BioBERT reward function.
    3. Normalise rewards within the group → advantages  A_i = (r_i - μ) / σ.
    4. Compute loss:
         L = -mean_i( A_i · log π_θ(y_i | x) )
             + β · mean_i( log π_θ(y_i | x) - log π_ref(y_i | x) )
    """
    G      = cfg.num_generations
    pad_id = processor.tokenizer.pad_token_id
    eos_id = processor.tokenizer.eos_token_id

    # ── 1. Generate G completions ─────────────────────────────────────────────
    policy.eval()
    comp_ids_list:  list[torch.Tensor] = []
    comp_text_list: list[str]          = []

    with torch.no_grad():
        for _ in range(G):
            gen_ids = policy.generate(
                **prompt_inputs,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=True,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            # Strip the prompt prefix; keep only new tokens
            comp = gen_ids[:, prompt_inputs["input_ids"].shape[1] :]   # (1, T_i)
            comp_ids_list.append(comp)
            comp_text_list.append(
                processor.tokenizer.decode(comp[0], skip_special_tokens=True)
            )

    policy.train()

    # ── 2. Rewards ────────────────────────────────────────────────────────────
    rewards = biobert_reward_fn(comp_text_list, ground_truth=[ground_truth] * G)
    R = torch.tensor(rewards, dtype=torch.float32, device=device)  # (G,)

    # ── 3. Group-relative advantages ─────────────────────────────────────────
    mean_R = R.mean()
    std_R  = R.std().clamp(min=1e-8)
    adv    = (R - mean_R) / std_R   # (G,)

    # ── 4. Policy-gradient + KL loss ─────────────────────────────────────────
    pg_terms: list[torch.Tensor] = []
    kl_terms: list[torch.Tensor] = []

    for i in range(G):
        comp = comp_ids_list[i]

        log_pi = compute_seq_log_prob(policy, prompt_inputs, comp)          # grad on

        with torch.no_grad():
            log_pi_ref = compute_seq_log_prob(ref, prompt_inputs, comp)     # no grad

        pg_terms.append(-adv[i] * log_pi.squeeze())
        kl_terms.append((log_pi - log_pi_ref).squeeze())

    pg_loss = torch.stack(pg_terms).mean()
    kl_loss = torch.stack(kl_terms).mean()
    loss    = pg_loss + cfg.beta * kl_loss

    return loss, {
        "loss":        loss.item(),
        "pg_loss":     pg_loss.item(),
        "kl":          kl_loss.item(),
        "reward_mean": mean_R.item(),
        "reward_std":  (std_R - 1e-8).item(),
    }


# ─── Training loop ────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Use MPS on Apple Silicon if available, fall back to CPU
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype  = torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")

    # ── Load model and processor ──────────────────────────────────────────────
    print(f"\nLoading {cfg.model_name} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
        device_map=str(device),
    )
    processor = AutoProcessor.from_pretrained(cfg.model_name)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Reference model: frozen deep copy of base (no LoRA) ──────────────────
    # Created BEFORE LoRA injection so it is the true base model.
    print("Creating frozen reference model ...")
    ref_model = copy.deepcopy(model)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    # ── Inject LoRA into the policy model ────────────────────────────────────
    n_lora = inject_lora(
        model,
        target_suffixes=list(cfg.lora_targets),
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        skip_prefixes=["model.vision_model"],   # vision encoder stays fully frozen
    )
    trainable, total = count_params(model)
    print(
        f"LoRA injected into {n_lora} layers  |  "
        f"trainable: {trainable:,} / {total:,}  ({100 * trainable / total:.2f}%)"
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds, val_ds = load_data(cfg)
    train_ds = train_ds.select(range(min(200, len(train_ds))))
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # ── Optimizer + cosine-with-warmup schedule ───────────────────────────────
    optimizer = AdamW(get_trainable_params(model), lr=cfg.lr, weight_decay=0.0)

    steps_per_epoch = math.ceil(len(train_ds) / cfg.grad_accum_steps)
    total_steps     = steps_per_epoch * cfg.num_epochs
    warmup_steps    = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * t)))   # cosine decay → 0

    scheduler = LambdaLR(optimizer, lr_lambda)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Print header ──────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("GRPO + LoRA  (raw PyTorch — no TRL / no PEFT)")
    print(f"  model           : {cfg.model_name}")
    print(f"  LoRA r / alpha  : {cfg.lora_r} / {cfg.lora_alpha}")
    print(f"  G (completions) : {cfg.num_generations}")
    print(f"  beta (KL coeff) : {cfg.beta}")
    print(f"  lr / epochs     : {cfg.lr} / {cfg.num_epochs}")
    print(f"  grad accum      : {cfg.grad_accum_steps}")
    print(f"  output          : {cfg.output_dir}")
    print("=" * 62 + "\n")

    # ── Training ──────────────────────────────────────────────────────────────
    global_step = 0
    running = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "reward_mean": 0.0}
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        for local_step, idx in enumerate(indices):
            example = train_ds[idx]

            # --- build processor inputs ---
            try:
                prompt_inputs = build_prompt_inputs(
                    example["prompt"], processor, device
                )
            except Exception as e:
                print(f"  [skip] input build failed (idx={idx}): {e}")
                continue

            # --- GRPO forward + compute loss ---
            try:
                loss, metrics = grpo_step(
                    model, ref_model,
                    prompt_inputs,
                    example["ground_truth"],
                    processor, cfg, device,
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM ] idx={idx} — clearing cache and skipping")
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise
            except Exception as e:
                print(f"  [skip] grpo_step failed (idx={idx}): {e}")
                optimizer.zero_grad()
                continue

            # --- gradient accumulation ---
            (loss / cfg.grad_accum_steps).backward()

            for k in running:
                running[k] += metrics[k] / cfg.grad_accum_steps

            if (local_step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    get_trainable_params(model), cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.log_steps == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"[ep {epoch + 1}/{cfg.num_epochs} | step {global_step:5d}]  "
                        f"loss={running['loss']:.4f}  "
                        f"pg={running['pg_loss']:.4f}  "
                        f"kl={running['kl']:.6f}  "
                        f"reward={running['reward_mean']:.4f}  "
                        f"lr={lr_now:.2e}"
                    )
                    running = {k: 0.0 for k in running}

                if global_step % cfg.save_steps == 0:
                    ckpt = os.path.join(cfg.output_dir, f"lora_step{global_step:05d}.pt")
                    save_lora_weights(model, ckpt)

    # ── Save final weights ────────────────────────────────────────────────────
    save_lora_weights(model, os.path.join(cfg.output_dir, "lora_final.pt"))
    processor.save_pretrained(cfg.output_dir)
    print(f"\nTraining complete. Saved to {cfg.output_dir}/")


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
