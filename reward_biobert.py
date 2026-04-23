"""
reward_biobert.py
BioBERT-based semantic similarity reward function for GRPO training.
Computes cosine similarity between generated captions and ground truth
using BioBERT embeddings.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Singleton to avoid reloading the model on every call
_biobert_model = None
_biobert_tokenizer = None
_device = None

BIOBERT_MODEL_NAME = "dmis-lab/biobert-large-cased-v1.1"


_load_attempted = False


def _get_biobert():
    """Lazily load BioBERT model and tokenizer (singleton)."""
    global _biobert_model, _biobert_tokenizer, _device, _load_attempted
    if _load_attempted:
        return _biobert_model, _biobert_tokenizer, _device
    _load_attempted = True

    print(f"Loading BioBERT from {BIOBERT_MODEL_NAME} ...")
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    # biobert-large-cased-v1.1 has no tokenizer.json; use the slow tokenizer
    _biobert_tokenizer = AutoTokenizer.from_pretrained(
        BIOBERT_MODEL_NAME, use_fast=False
    )
    _biobert_model = AutoModel.from_pretrained(BIOBERT_MODEL_NAME).to(_device)
    _biobert_model.eval()
    print(f"BioBERT loaded on {_device}")
    return _biobert_model, _biobert_tokenizer, _device


def _encode_texts(texts: list[str]) -> torch.Tensor:
    """Encode a list of texts into BioBERT CLS embeddings."""
    model, tokenizer, device = _get_biobert()
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use CLS token embedding (first token)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings


def biobert_reward_fn(completions, ground_truth=None):
    """
    Compute BioBERT cosine similarity reward for GRPO.

    Args:
        completions: list of generated strings (or list of message dicts)
        ground_truth: list of reference caption strings

    Returns:
        list[float]: reward scores in [0, 1] for each completion
    """
    if ground_truth is None:
        return [0.0] * len(completions)

    # Extract text from completions (handle both str and message-dict formats)
    pred_texts = []
    for comp in completions:
        if isinstance(comp, list):
            # Message format: [{"role": "assistant", "content": "..."}]
            text = comp[0].get("content", "") if comp else ""
        elif isinstance(comp, dict):
            text = comp.get("content", "")
        else:
            text = str(comp)
        pred_texts.append(text.strip())

    gt_texts = [str(gt).strip() for gt in ground_truth]

    if _biobert_model is None:
        return [0.0] * len(completions)

    # Encode predictions and ground truths
    pred_emb = _encode_texts(pred_texts)  # (N, 768)
    gt_emb = _encode_texts(gt_texts)  # (N, 768)

    # Cosine similarity per pair
    similarities = F.cosine_similarity(pred_emb, gt_emb, dim=-1)  # (N,)

    # Map from [-1, 1] to [0, 1] — clamp negatives to 0
    rewards = torch.clamp(similarities, min=0.0).cpu().tolist()

    return rewards


if __name__ == "__main__":
    # Quick sanity check
    preds = [
        "Normal chest radiograph with no acute cardiopulmonary abnormality.",
        "This is a cat sitting on a table.",
        "Bilateral pleural effusion with cardiomegaly.",
    ]
    refs = [
        "Normal chest x-ray. No acute findings.",
        "Normal chest x-ray. No acute findings.",
        "Large bilateral pleural effusions and enlarged cardiac silhouette.",
    ]
    scores = biobert_reward_fn(preds, ground_truth=refs)
    for p, r, s in zip(preds, refs, scores):
        print(f"Score: {s:.4f}")
        print(f"  Pred: {p}")
        print(f"  Ref:  {r}\n")
