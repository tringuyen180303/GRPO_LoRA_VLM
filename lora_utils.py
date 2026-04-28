"""
lora_utils.py
Manual LoRA injection — replaces peft entirely.

Public API
----------
inject_lora(model, target_suffixes, r, lora_alpha, lora_dropout, skip_prefixes)
get_trainable_params(model)
count_params(model)
save_lora_weights(model, path)
load_lora_weights(model, path)
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a frozen base weight and a
    trainable low-rank adapter (LoRA).

    Forward:  out = base(x)  +  (dropout(x) @ A.T @ B.T) * (alpha / r)

    B is zero-initialised so the adapter contributes nothing at step 0,
    making the model identical to the base at the start of training.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 32,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base = base
        self.r = r
        self.scale = lora_alpha / r

        d_in = base.in_features
        d_out = base.out_features
        dev = base.weight.device

        self.lora_A = nn.Parameter(torch.empty(r, d_in, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=dev))   # B=0 → Δ=0 at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Freeze base weights
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale

    def extra_repr(self) -> str:
        return (
            f"in={self.base.in_features}, out={self.base.out_features}, "
            f"r={self.r}, scale={self.scale:.2f}"
        )


def inject_lora(
    model: nn.Module,
    target_suffixes: list[str],
    r: int = 32,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
    skip_prefixes: list[str] | None = None,
) -> int:
    """
    1. Freeze every parameter in the model.
    2. Replace each nn.Linear whose attribute name (last dotted segment) is in
       `target_suffixes` with a LoRALinear — unless its full dotted path starts
       with any of `skip_prefixes`.

    Returns the number of layers replaced.
    """
    skip_prefixes = skip_prefixes or []

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    # Collect replacements without mutating during named_modules() iteration
    to_replace: list[tuple[str, str, nn.Linear]] = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(full_name.startswith(pfx) for pfx in skip_prefixes):
            continue
        if "." not in full_name:
            continue                                      # top-level module, skip
        short_name = full_name.rsplit(".", 1)[-1]
        if short_name not in target_suffixes:
            continue
        parent_path, child_name = full_name.rsplit(".", 1)
        to_replace.append((parent_path, child_name, module))

    for parent_path, child_name, linear in to_replace:
        parent = model.get_submodule(parent_path)
        setattr(
            parent,
            child_name,
            LoRALinear(linear, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
        )

    return len(to_replace)


def get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    """Return all parameters with requires_grad=True (the LoRA matrices)."""
    return [p for p in model.parameters() if p.requires_grad]


def count_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_count, total_count)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA adapter tensors (lora_A and lora_B)."""
    lora_sd = {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    torch.save(lora_sd, path)
    print(f"  Saved {len(lora_sd)} LoRA tensors → {path}")


def load_lora_weights(model: nn.Module, path: str, device: str = "cpu") -> None:
    """
    Load previously saved LoRA adapter tensors into a model that already has
    LoRA injected (e.g. after calling inject_lora on the same architecture).
    """
    state = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"  Loaded LoRA weights from {path} | "
        f"missing: {len(missing)}, unexpected: {len(unexpected)}"
    )
