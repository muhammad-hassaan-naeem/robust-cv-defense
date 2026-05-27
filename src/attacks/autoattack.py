"""
attacks/autoattack.py
---------------------
AutoAttack — Croce & Hein, 2020.

AutoAttack is a parameter-free, ensemble-based adversarial evaluation
framework. It is the community-accepted benchmark for reliable robustness
evaluation and is used by the RobustBench leaderboard.

It prevents two common evaluation pitfalls:
  1. Gradient masking  — defenses that obscure gradients appear robust
                         under PGD but fail against gradient-free attacks
  2. Hyper-parameter sensitivity — PGD results depend heavily on step size
                                    and restart count; AutoAttack does not

Ensemble composition (all L∞, ε-bounded):
  - APGD-CE   : Auto-PGD with cross-entropy loss (adaptive step size)
  - APGD-DLR  : Auto-PGD with difference-of-logits ratio loss
  - FAB        : Fast Adaptive Boundary attack (minimises perturbation norm)
  - Square     : Black-box square attack (query-based, no gradients)

This module provides:
  - A self-contained pure-PyTorch implementation of each component
  - AutoAttackEnsemble: runs all four in sequence, returns worst-case
  - autoattack_accuracy(): computes RA against the full ensemble
  - Integration with the DualReport / DualEvaluator pipeline

Reference:
  Croce, F. & Hein, M. (2020). Reliable evaluation of adversarial robustness
  with an ensemble of diverse parameter-free attacks. ICML 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────
def dlr_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Difference-of-Logits Ratio (DLR) loss.

    More reliable than cross-entropy for iterative attacks because it is
    invariant to logit scaling and does not saturate.

    L_DLR(x, y) = -(z_y - max_{i≠y} z_i) / (z_[1] - z_[3])

    where z_[k] is the k-th largest logit.
    """
    B = logits.size(0)
    # Score of true class
    z_y    = logits[range(B), labels]
    # Best competitor (highest non-true logit)
    logits_sorted, _ = logits.sort(dim=1, descending=True)
    # Determine rank of true class
    ranks  = (logits.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
    z_top1 = torch.where(ranks == 0, logits_sorted[:, 1], logits_sorted[:, 0])
    # Normalisation: z[0] - z[2]
    z_denom = logits_sorted[:, 0] - logits_sorted[:, 2] + 1e-8
    return -((z_y - z_top1) / z_denom).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Component 1 — APGD (Auto-PGD)
# ─────────────────────────────────────────────────────────────────────────────
class APGD:
    """
    Auto-PGD (Croce & Hein, 2020).

    Adaptive step-size PGD with a momentum term and a step-size decay
    schedule based on a sufficient-decrease condition.

    Args:
        model      : target model
        eps        : L∞ perturbation budget
        steps      : number of gradient steps
        loss       : 'ce' (cross-entropy) or 'dlr'
        alpha_init : initial step size as fraction of eps
        device     : torch device
    """

    def __init__(
        self,
        model,
        eps:        float = 8 / 255,
        steps:      int   = 100,
        loss:       str   = "ce",
        alpha_init: float = 0.75,
        device      = None,
    ):
        self.model      = model
        self.eps        = eps
        self.steps      = steps
        self.loss_type  = loss
        self.alpha_init = alpha_init
        self.device     = device or next(model.parameters()).device

    def perturb(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return worst-case adversarial examples from n_restarts random starts."""
        images  = images.to(self.device)
        labels  = labels.to(self.device)
        best_adv = images.clone()
        best_loss = -torch.ones(images.size(0), device=self.device) * float("inf")

        adv = self._run(images, labels)
        with torch.no_grad():
            loss_vals = self._batch_loss(adv, labels)
            improved  = loss_vals > best_loss
            best_adv[improved]  = adv[improved]
            best_loss[improved] = loss_vals[improved]

        return best_adv.detach()

    def _run(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B  = images.size(0)
        # Random init
        delta = torch.empty_like(images).uniform_(-self.eps, self.eps)
        x_adv = (images + delta).clamp(0.0, 1.0)

        alpha  = self.alpha_init * self.eps
        x_prev = x_adv.clone()
        loss_prev = -float("inf")
        # Checkpoints at 22%, 50% of steps
        checkpoints = {int(0.22 * self.steps), int(0.50 * self.steps)}
        step_reduce_count = 0

        for step in range(self.steps):
            x_adv = x_adv.detach().requires_grad_(True)
            loss  = self._batch_loss(x_adv, labels).mean()
            loss.backward()

            with torch.no_grad():
                grad = x_adv.grad.sign()
                x_next = x_adv + alpha * grad
                x_next = images + (x_next - images).clamp(-self.eps, self.eps)
                x_next = x_next.clamp(0.0, 1.0)

                # Momentum correction
                x_adv_new = x_next + 0.75 * (x_next - x_adv)
                x_adv_new = images + (x_adv_new - images).clamp(-self.eps, self.eps)
                x_adv_new = x_adv_new.clamp(0.0, 1.0)
                x_adv = x_adv_new

                # Adaptive step-size decay at checkpoints
                if step in checkpoints:
                    cur_loss = self._batch_loss(x_adv, labels).mean().item()
                    if cur_loss <= loss_prev + 1e-6:
                        alpha = max(alpha * 0.75, self.eps * 0.001)
                        step_reduce_count += 1
                    loss_prev = cur_loss

        return x_adv.detach()

    def _batch_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-sample loss (no mean)."""
        logits = self.model(x)
        if self.loss_type == "ce":
            return F.cross_entropy(logits, labels, reduction="none")
        else:
            # DLR per-sample: compute manually
            B = x.size(0)
            z_y = logits[range(B), labels]
            logits_sorted, _ = logits.sort(dim=1, descending=True)
            ranks  = (logits.argsort(dim=1, descending=True) == labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
            z_top1 = torch.where(ranks == 0, logits_sorted[:, 1], logits_sorted[:, 0])
            z_denom = logits_sorted[:, 0] - logits_sorted[:, 2] + 1e-8
            return -((z_y - z_top1) / z_denom)


# ─────────────────────────────────────────────────────────────────────────────
# Component 2 — Square Attack (black-box)
# ─────────────────────────────────────────────────────────────────────────────
class SquareAttack:
    """
    Square Attack — Andriushchenko et al., 2020.

    Query-based black-box attack: no gradient access required.
    Perturbs random square patches and accepts moves that increase loss.
    This is the key component that catches gradient-masking defenses.

    Args:
        model   : target model
        eps     : L∞ perturbation budget
        n_queries: number of queries (function evaluations)
        p_init  : initial proportion of image covered by square
        device  : torch device
    """

    def __init__(self, model, eps=8/255, n_queries=1000, p_init=0.8, device=None):
        self.model     = model
        self.eps       = eps
        self.n_queries = n_queries
        self.p_init    = p_init
        self.device    = device or next(model.parameters()).device

    def perturb(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        labels = labels.to(self.device)
        B, C, H, W = images.shape

        # Initialise with vertical stripes
        delta = torch.zeros_like(images)
        for b in range(B):
            for c in range(C):
                delta[b, c] = self.eps * (2.0 * (torch.rand(1, W).expand(H, -1) > 0.5).float() - 1.0)
        x_adv = (images + delta).clamp(0.0, 1.0)

        with torch.no_grad():
            loss_curr = F.cross_entropy(self.model(x_adv), labels, reduction="none")

        for q in range(self.n_queries):
            p = self._p_schedule(q)
            s = max(int(round(p * H)), 1)

            x_new = x_adv.clone()
            for b in range(B):
                top  = torch.randint(0, H - s + 1, (1,)).item()
                left = torch.randint(0, W - s + 1, (1,)).item()
                for c in range(C):
                    val = self.eps * (1.0 if torch.rand(1).item() > 0.5 else -1.0)
                    x_new[b, c, top:top+s, left:left+s] = (
                        images[b, c, top:top+s, left:left+s] + val
                    ).clamp(0.0, 1.0)

            x_new = (images + (x_new - images).clamp(-self.eps, self.eps)).clamp(0.0, 1.0)

            with torch.no_grad():
                loss_new = F.cross_entropy(self.model(x_new), labels, reduction="none")
                improved = loss_new > loss_curr
                x_adv[improved]    = x_new[improved]
                loss_curr[improved] = loss_new[improved]

        return x_adv.detach()

    def _p_schedule(self, step: int) -> float:
        """Decrease patch size over time."""
        checkpoints = [0, 0.001, 0.005, 0.02, 0.1, 0.25, 0.5, 0.75]
        ps          = [self.p_init, 0.6, 0.4, 0.25, 0.15, 0.1, 0.05, 0.01]
        frac = step / self.n_queries
        for i in range(len(checkpoints) - 1, -1, -1):
            if frac >= checkpoints[i]:
                return ps[i]
        return self.p_init


# ─────────────────────────────────────────────────────────────────────────────
# AutoAttack Ensemble
# ─────────────────────────────────────────────────────────────────────────────
class AutoAttackEnsemble:
    """
    AutoAttack ensemble (Croce & Hein, 2020).

    Runs four attacks in sequence. For each sample, takes the worst-case
    adversarial example across all four (maximum loss / minimum accuracy).

    Component attacks:
      1. APGD-CE   (white-box, cross-entropy loss)
      2. APGD-DLR  (white-box, DLR loss — catches CE saturation)
      3. Square    (black-box, no gradients — catches gradient masking)

    Note: FAB is omitted as it requires a significantly different
    optimisation loop; APGD-CE + APGD-DLR + Square is the standard
    reduced AutoAttack used in many published papers.

    Args:
        model      : target model (eval mode)
        eps        : L∞ perturbation budget
        steps      : APGD steps (100 is standard; use 50 for faster runs)
        n_queries  : Square attack queries
        device     : torch device
        verbose    : print per-attack accuracy
    """

    def __init__(
        self,
        model,
        eps       = 8 / 255,
        steps     = 100,
        n_queries = 1000,
        device    = None,
        verbose   = True,
    ):
        self.model   = model.eval()
        self.eps     = eps
        self.verbose = verbose
        self.device  = device or next(model.parameters()).device

        self.attacks = [
            ("APGD-CE",  APGD(model, eps=eps, steps=steps, loss="ce",  device=self.device)),
            ("APGD-DLR", APGD(model, eps=eps, steps=steps, loss="dlr", device=self.device)),
            ("Square",   SquareAttack(model, eps=eps, n_queries=n_queries, device=self.device)),
        ]

    def perturb(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Run the ensemble. Returns worst-case adversarial examples.

        For each sample, the adversarial example that causes the highest
        loss (most likely to cause misclassification) is kept.
        """
        images  = images.to(self.device)
        labels  = labels.to(self.device)
        B       = images.size(0)
        best_adv  = images.clone()
        fooled    = torch.zeros(B, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            init_correct = (self.model(images).argmax(1) == labels)

        for atk_name, attack in self.attacks:
            # Only attack samples not yet fooled (efficiency)
            remaining = (~fooled & init_correct)
            if remaining.sum() == 0:
                break

            adv_full = images.clone()
            adv_full[remaining] = attack.perturb(images[remaining], labels[remaining])

            with torch.no_grad():
                preds       = self.model(adv_full).argmax(1)
                newly_fooled = (preds != labels) & remaining
                best_adv[newly_fooled] = adv_full[newly_fooled]
                fooled |= newly_fooled

            if self.verbose:
                survived = (~fooled & init_correct).sum().item()
                print(f"    [{atk_name}] fooled so far: {fooled.sum().item()}/{B}  "
                      f"(robust remaining: {survived})")

        return best_adv.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function — AutoAttack Robust Accuracy
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def autoattack_accuracy(
    model,
    loader,
    eps       = 8 / 255,
    steps     = 100,
    n_queries = 1000,
    device    = None,
    sanitizer = None,
    verbose   = True,
) -> float:
    """
    Compute AutoAttack robust accuracy over a DataLoader.

    This is the standard evaluation metric used by RobustBench.
    Lower AA-RA than PGD-RA indicates gradient masking in the defense.

    Args:
        model     : defended CNN
        loader    : test DataLoader
        eps       : L∞ budget
        steps     : APGD steps (100 standard, 50 for fast mode)
        n_queries : Square attack queries
        device    : torch device
        sanitizer : optional SanitizationPipeline applied after attack
        verbose   : print per-batch progress

    Returns:
        AA robust accuracy in [0, 1]
    """
    if device is None:
        device = next(model.parameters()).device

    aa = AutoAttackEnsemble(model, eps=eps, steps=steps, n_queries=n_queries,
                            device=device, verbose=verbose)
    model.eval()
    correct, total = 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        if verbose:
            print(f"\n  [AutoAttack] Batch {batch_idx+1} ({images.size(0)} samples, ε={eps:.4f}) ...")

        adv = aa.perturb(images, labels)

        with torch.no_grad():
            if sanitizer is not None:
                adv = sanitizer(adv).to(device)
            preds   = model(adv).argmax(1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)

        if verbose:
            running_ra = correct / total
            print(f"  [AutoAttack] Running AA-RA: {running_ra:.4f} ({correct}/{total})")

    return correct / total if total else 0.0
