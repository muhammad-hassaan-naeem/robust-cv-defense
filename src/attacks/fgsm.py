"""
attacks/fgsm.py
---------------
Fast Gradient Sign Method (FGSM) — Goodfellow et al., 2014.

Equation from the proposal:
    x' = x + ε · sign(∇ₓ L(θ, x, y))

White-box, single-step, L∞-norm attack.
"""

import torch
import torch.nn as nn


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 8 / 255,
    loss_fn: nn.Module | None = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Generate FGSM adversarial examples.

    Args:
        model    : target CNN (in eval mode)
        images   : clean inputs,  shape (B, C, H, W), values in [0, 1]
        labels   : ground-truth labels, shape (B,)
        eps      : perturbation magnitude in pixel space (default 8/255 ≈ 0.031)
        loss_fn  : loss function; defaults to CrossEntropyLoss
        clip_min : lower bound of valid pixel range
        clip_max : upper bound of valid pixel range

    Returns:
        adv_images : adversarial inputs, same shape as `images`
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    images  = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss    = loss_fn(outputs, labels)

    model.zero_grad()
    loss.backward()

    # ∇ₓ L(θ, x, y)
    grad_sign   = images.grad.data.sign()
    adv_images  = images.detach() + eps * grad_sign
    adv_images  = adv_images.clamp(clip_min, clip_max)

    return adv_images


def fgsm_batch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    eps: float = 8 / 255,
    device: torch.device | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Apply FGSM to an entire DataLoader, returning a list of
    (adv_images, labels) tuples.

    Args:
        model      : target CNN
        dataloader : standard DataLoader
        eps        : perturbation magnitude
        device     : torch device; auto-detected if None

    Returns:
        List of (adv_batch, labels) tuples
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    corpus = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        adv = fgsm_attack(model, images, labels, eps=eps)
        corpus.append((adv.cpu(), labels.cpu()))

    return corpus
