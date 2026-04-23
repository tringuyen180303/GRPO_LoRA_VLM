"""
train_grpo_raw.py
GRPO + LoRA fine-tuning of Qwen2.5-VL-3B on Radiology_mini.
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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
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
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    output_dir: str = "./xray_grpo_lora_raw"

    # LoRA — only language-model layers; vision encoder is always frozen
    lora_r: int = 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",       # MLP
    )

    # GRPO
    num_generations: int = 2       # ↓ was 4; cuts generation memory in half
    beta: float = 0.001            # KL-penalty coefficient

    # Training loop
    num_epochs: int = 3
    lr: float = 1e-5
    warmup_ratio: float = 0.1
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0

    # Generation
    max_new_tokens: int = 64       # ↓ was 128
    temperature: float = 0.9
    top_p: float = 0.9

    # Misc
    bf16: bool = True
    log_steps: int = 5
    save_steps: int = 50
    seed: int = 42
    data_dir: str = "./data"

    # Memory
    offload_ref_to_cpu: bool = True   # keep ref model on CPU, move per-call
    gradient_checkpointing: bool = True


# ─── Memory helpers ───────────────────────────────────────────────────────────────

def mem_stats(device: torch.device) -> str:
    alloc   = torch.cuda.memory_allocated(device)  / 1024**3
    reserved = torch.cuda.memory_reserved(device)  / 1024**3
    return f"GPU mem: {alloc:.1f}/{reserved:.1f} GB (alloc/reserved)"


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
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            images.append(img)
    return images


def build_prompt_inputs(
    messages: list,
    processor: AutoProcessor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
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
    target_device: torch.device,
) -> torch.Tensor:
    """
    Return  Σ_t  log π(token_t | context)  summed over the T completion tokens.
    prompt_inputs and completion_ids are moved to whatever device `model` is on,
    then the result is returned on target_device.
    """
    # Infer model device from first parameter
    model_device = next(model.parameters()).device

    # Move inputs to model's device
    p_inputs = {k: v.to(model_device) for k, v in prompt_inputs.items()}
    comp     = completion_ids.to(model_device)

    L = p_inputs["input_ids"].shape[1]
    T = comp.shape[1]

    full_ids  = torch.cat([p_inputs["input_ids"],  comp], dim=1)
    full_mask = torch.cat([p_inputs["attention_mask"], torch.ones_like(comp)], dim=1)

    kwargs: dict = {"input_ids": full_ids, "attention_mask": full_mask}
    if "pixel_values"   in p_inputs:
        kwargs["pixel_values"]   = p_inputs["pixel_values"]
    if "image_grid_thw" in p_inputs:
        kwargs["image_grid_thw"] = p_inputs["image_grid_thw"]

    outputs     = model(**kwargs)
    comp_logits = outputs.logits[:, L - 1 : L + T - 1, :]
    log_probs   = F.log_softmax(comp_logits, dim=-1)
    token_lp    = log_probs.gather(2, comp.unsqueeze(-1)).squeeze(-1)
    return token_lp.sum(dim=-1).to(target_device)


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
    G      = cfg.num_generations
    pad_id = processor.tokenizer.pad_token_id
    eos_id = processor.tokenizer.eos_token_id

    # ── 1. Generate G completions (no grad) ──────────────────────────────────
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
            comp = gen_ids[:, prompt_inputs["input_ids"].shape[1]:].cpu()  # store on CPU
            comp_ids_list.append(comp)
            comp_text_list.append(
                processor.tokenizer.decode(comp[0], skip_special_tokens=True)
            )
            del gen_ids
            torch.cuda.empty_cache()

    policy.train()

    # ── 2. Rewards ────────────────────────────────────────────────────────────
    rewards = biobert_reward_fn(comp_text_list, ground_truth=[ground_truth] * G)
    R   = torch.tensor(rewards, dtype=torch.float32, device=device)
    mean_R = R.mean()
    std_R  = R.std().clamp(min=1e-8)
    adv    = (R - mean_R) / std_R

    # ── 3. Policy-gradient + KL loss (one completion at a time) ───────────────
    pg_terms: list[torch.Tensor] = []
    kl_terms: list[torch.Tensor] = []

    for i in range(G):
        comp = comp_ids_list[i].to(device)   # move to GPU just for this pass

        log_pi = compute_seq_log_prob(policy, prompt_inputs, comp, device)

        with torch.no_grad():
            log_pi_ref = compute_seq_log_prob(ref, prompt_inputs, comp, device)

        pg_terms.append(-adv[i] * log_pi.squeeze())
        kl_terms.append((log_pi - log_pi_ref).squeeze())

        del comp
        torch.cuda.empty_cache()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if cfg.bf16 and device.type == "cuda" else torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")

    torch.cuda.reset_peak_memory_stats(device)

    # ── Load policy model ─────────────────────────────────────────────────────
    print(f"\nLoading {cfg.model_name} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
    )
    model.to(device)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    processor = AutoProcessor.from_pretrained(cfg.model_name)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ── Reference model ───────────────────────────────────────────────────────
    print("Creating frozen reference model ...")
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,
    )
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    if cfg.offload_ref_to_cpu:
        ref_model.to("cpu")   # stays on CPU; compute_seq_log_prob moves inputs automatically
        print("Reference model offloaded to CPU.")
    else:
        ref_model.to(device)

    print(mem_stats(device))

    # ── Inject LoRA ───────────────────────────────────────────────────────────
    n_lora = inject_lora(
        model,
        target_suffixes=list(cfg.lora_targets),
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        skip_prefixes=["visual"],
    )
    model.to(device=device, dtype=dtype)   # sync LoRA params to correct device/dtype

    trainable, total = count_params(model)
    print(
        f"LoRA injected into {n_lora} layers  |  "
        f"trainable: {trainable:,} / {total:,}  ({100 * trainable / total:.2f}%)"
    )
    print(mem_stats(device))

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds, val_ds = load_data(cfg)
    print(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # ── Optimizer + schedule ──────────────────────────────────────────────────
    optimizer = AdamW(get_trainable_params(model), lr=cfg.lr, weight_decay=0.0)

    steps_per_epoch = math.ceil(len(train_ds) / cfg.grad_accum_steps)
    total_steps     = steps_per_epoch * cfg.num_epochs
    warmup_steps    = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * t)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("GRPO + LoRA  (raw PyTorch — no TRL / no PEFT)")
    print(f"  model             : {cfg.model_name}")
    print(f"  LoRA r / alpha    : {cfg.lora_r} / {cfg.lora_alpha}")
    print(f"  G (completions)   : {cfg.num_generations}")
    print(f"  beta (KL coeff)   : {cfg.beta}")
    print(f"  lr / epochs       : {cfg.lr} / {cfg.num_epochs}")
    print(f"  grad accum        : {cfg.grad_accum_steps}")
    print(f"  ref on CPU        : {cfg.offload_ref_to_cpu}")
    print(f"  grad checkpointing: {cfg.gradient_checkpointing}")
    print(f"  output            : {cfg.output_dir}")
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

            try:
                prompt_inputs = build_prompt_inputs(
                    example["prompt"], processor, device
                )
            except Exception as e:
                print(f"  [skip] input build failed (idx={idx}): {e}")
                continue

            try:
                loss, metrics = grpo_step(
                    model, ref_model,
                    prompt_inputs,
                    example["ground_truth"],
                    processor, cfg, device,
                )
            except torch.cuda.OutOfMemoryError:
                alloc = torch.cuda.memory_allocated(device) / 1024**3
                res   = torch.cuda.memory_reserved(device)  / 1024**3
                print(f"  [OOM ] idx={idx} — alloc={alloc:.1f}GB reserved={res:.1f}GB")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            except Exception as e:
                print(f"  [skip] grpo_step failed (idx={idx}): {e}")
                optimizer.zero_grad()
                continue

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
                    peak   = torch.cuda.max_memory_allocated(device) / 1024**3
                    print(
                        f"[ep {epoch + 1}/{cfg.num_epochs} | step {global_step:5d}]  "
                        f"loss={running['loss']:.4f}  "
                        f"pg={running['pg_loss']:.4f}  "
                        f"kl={running['kl']:.6f}  "
                        f"reward={running['reward_mean']:.4f}  "
                        f"lr={lr_now:.2e}  "
                        f"peak_mem={peak:.1f}GB"
                    )
                    running = {k: 0.0 for k in running}
                    torch.cuda.reset_peak_memory_stats(device)

                if global_step % cfg.save_steps == 0:
                    ckpt = os.path.join(cfg.output_dir, f"lora_step{global_step:05d}.pt")
                    save_lora_weights(model, ckpt)

    save_lora_weights(model, os.path.join(cfg.output_dir, "lora_final.pt"))
    processor.save_pretrained(cfg.output_dir)
    print(f"\nTraining complete. Saved to {cfg.output_dir}/")


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
