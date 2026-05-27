"""
attacks/corpus.py
-----------------
Phase I — Threat Modeling.

Generates and saves the frozen attack corpus used for all evaluation
across Phases II–IV.  Running this once guarantees that all defense
configurations are compared against identical adversarial inputs.

Attacks generated:
  1. FGSM (L∞, ε=8/255)         — baseline single-step
  2. PGD-7 (L∞, ε=8/255)        — strong white-box
  3. PGD-20 (L∞, ε=8/255)       — stronger white-box
  4. PGD-7 (L2, ε=0.5)          — L2-norm variant
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime

from attacks.fgsm import fgsm_attack
from attacks.pgd  import pgd_attack


# ──────────────────────────────────────────────────────────────────────────────
ATTACK_CONFIGS = [
    dict(name="FGSM",    norm="Linf", eps=8/255, alpha=None, steps=1),
    dict(name="PGD-7",   norm="Linf", eps=8/255, alpha=2/255, steps=7),
    dict(name="PGD-20",  norm="Linf", eps=8/255, alpha=2/255, steps=20),
    dict(name="PGD-7-L2",norm="L2",   eps=0.5,   alpha=0.1,   steps=7),
]


def generate_corpus(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    save_dir: str | Path,
    device: torch.device | None = None,
    configs: list[dict] | None = None,
) -> dict:
    """
    Generate the full attack corpus and save to disk.

    Args:
        model      : target CNN (frozen — eval mode only)
        dataloader : test DataLoader
        save_dir   : directory to save .pt files + metadata JSON
        device     : torch device
        configs    : list of attack config dicts; defaults to ATTACK_CONFIGS

    Returns:
        metadata dict describing the corpus
    """
    if device is None:
        device = next(model.parameters()).device
    if configs is None:
        configs = ATTACK_CONFIGS

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "attacks": [],
        "n_samples": 0,
    }

    loss_fn = torch.nn.CrossEntropyLoss()
    print(f"\n[Phase I] Generating attack corpus → {save_dir}")

    for cfg in configs:
        name  = cfg["name"]
        norm  = cfg["norm"]
        eps   = cfg["eps"]
        alpha = cfg.get("alpha", eps / 4)
        steps = cfg["steps"]

        print(f"  ▸ {name:12s}  norm={norm}  ε={eps:.4f}  steps={steps} ...", end="", flush=True)

        adv_batches, label_batches = [], []

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            if steps == 1:
                adv = fgsm_attack(model, images, labels, eps=eps, loss_fn=loss_fn)
            else:
                adv = pgd_attack(
                    model, images, labels,
                    eps=eps, alpha=alpha, steps=steps,
                    norm=norm, random_start=True, loss_fn=loss_fn,
                )

            adv_batches.append(adv.cpu())
            label_batches.append(labels.cpu())

        adv_tensor    = torch.cat(adv_batches,   dim=0)
        label_tensor  = torch.cat(label_batches, dim=0)
        filename      = f"corpus_{name.replace('-', '_').lower()}.pt"
        torch.save({"images": adv_tensor, "labels": label_tensor}, save_dir / filename)

        metadata["attacks"].append({
            "name":     name,
            "norm":     norm,
            "eps":      eps,
            "alpha":    alpha,
            "steps":    steps,
            "filename": filename,
            "n_samples": len(adv_tensor),
        })
        metadata["n_samples"] = len(adv_tensor)
        print(f" saved {len(adv_tensor)} samples → {filename}")

    # Save metadata
    with open(save_dir / "corpus_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Phase I] Done. Corpus metadata → {save_dir / 'corpus_metadata.json'}\n")
    return metadata


def load_corpus(save_dir: str | Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Load a previously generated corpus from disk.

    Returns:
        Dict mapping attack name → (adv_images, labels)
    """
    save_dir = Path(save_dir)
    with open(save_dir / "corpus_metadata.json") as f:
        metadata = json.load(f)

    corpus = {}
    for atk in metadata["attacks"]:
        data = torch.load(save_dir / atk["filename"], map_location="cpu", weights_only=True)
        corpus[atk["name"]] = (data["images"], data["labels"])

    return corpus
