"""
attacks/pgd.py
--------------
Projected Gradient Descent (PGD) — Madry et al., 2018.

The strongest first-order adversarial attack.
Iteratively applies FGSM steps and projects back to the ε-ball.

Supports:
  - L∞-norm attacks  (default)
  - L2-norm attacks
  - Random restarts
"""

import torch
import torch.nn as nn


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 7,
    norm: str = "Linf",
    random_start: bool = True,
    loss_fn: nn.Module | None = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Generate PGD adversarial examples.

    Args:
        model        : target CNN (eval mode)
        images       : clean inputs  (B, C, H, W) in [0, 1]
        labels       : ground-truth labels  (B,)
        eps          : perturbation ball radius
        alpha        : step size per iteration
        steps        : number of gradient steps (PGD-7 = steps=7)
        norm         : 'Linf' | 'L2'
        random_start : initialise from random point in ε-ball (recommended)
        loss_fn      : defaults to CrossEntropyLoss
        clip_min     : lower pixel bound
        clip_max     : upper pixel bound

    Returns:
        adv_images : adversarial batch, same shape as images
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    images = images.clone().detach()

    # ── Random initialisation within ε-ball ──────────────────────────────────
    if random_start:
        if norm == "Linf":
            delta = torch.empty_like(images).uniform_(-eps, eps)
        else:  # L2
            delta = torch.randn_like(images)
            delta = delta / (delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-9) * eps
        adv = (images + delta).clamp(clip_min, clip_max)
    else:
        adv = images.clone()

    # ── Iterative gradient steps ──────────────────────────────────────────────
    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = loss_fn(model(adv), labels)
        loss.backward()

        with torch.no_grad():
            grad = adv.grad

            if norm == "Linf":
                adv = adv + alpha * grad.sign()
                # Project back to ε-ball around original image
                adv = images + (adv - images).clamp(-eps, eps)

            else:  # L2
                # Normalise gradient
                g_norm = grad.view(grad.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                adv = adv + alpha * grad / (g_norm + 1e-9)
                # L2-ball projection
                delta = adv - images
                d_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1).view(-1, 1, 1, 1)
                delta  = delta / d_norm.clamp(min=1.0) * eps
                adv = images + delta

            adv = adv.clamp(clip_min, clip_max)

    return adv.detach()


def pgd_batch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 7,
    norm: str = "Linf",
    device: torch.device | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply PGD to an entire DataLoader.

    Returns:
        List of (adv_batch, labels) tuples
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    corpus = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        adv = pgd_attack(model, images, labels, eps=eps, alpha=alpha, steps=steps, norm=norm)
        corpus.append((adv.cpu(), labels.cpu()))

    return corpus
