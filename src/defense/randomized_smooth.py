"""
defense/randomized_smooth.py
-----------------------------
Layer 3 — Randomized Smoothing  (Cohen et al., 2019)

Provides certified L₂ robustness: for a given input x, computes a radius δ
such that no perturbation η with ‖η‖₂ < δ can change the smoothed classifier's
prediction.

Smoothed classifier:
    g(x) = argmax_c  P[f(x + ε) = c],   ε ~ N(0, σ²I)

Certification via the Neyman-Pearson lemma:
    If P[f(x + ε) = c*] ≥ p̄  and  P[f(x + ε) = c_runner] ≤ p_under,
    then  δ = (σ/2) · (Φ⁻¹(p̄) - Φ⁻¹(p_under))

Where Φ is the standard normal CDF.

Reference:
    Cohen, J., Rosenfeld, E., & Kolter, J. Z. (2019).
    Certified adversarial robustness via randomized smoothing.
    ICML 2019.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import norm as scipy_norm
from scipy.stats import binomtest


ABSTAIN = -1   # sentinel: smoothed classifier abstains (confidence too low)


class RandomizedSmoother:
    """
    Wraps a base classifier with Randomized Smoothing certification.

    Args:
        base_model  : the underlying CNN (f in the proposal)
        sigma       : noise level for Gaussian smoothing (σ)
        device      : torch device
        n_samples   : number of Monte-Carlo samples for prediction
        n_samples_cert : number of samples for certification (larger = tighter)
        alpha       : failure probability for certification (default 0.001)
    """

    def __init__(
        self,
        base_model:     nn.Module,
        sigma:          float = 0.25,
        device:         torch.device | None = None,
        n_samples:      int   = 100,
        n_samples_cert: int   = 1000,
        alpha:          float = 0.001,
    ):
        self.model          = base_model
        self.sigma          = sigma
        self.device         = device or next(base_model.parameters()).device
        self.n_samples      = n_samples
        self.n_samples_cert = n_samples_cert
        self.alpha          = alpha

        self.model.eval()

    # ── Prediction ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        x:         torch.Tensor,
        n_samples: int | None = None,
        batch_sz:  int        = 64,
    ) -> torch.Tensor:
        """
        Smoothed prediction: majority vote over noisy copies.

        Args:
            x         : single image (C, H, W) or batch (B, C, H, W)
            n_samples : MC samples (defaults to self.n_samples)
            batch_sz  : internal batch size for memory efficiency

        Returns:
            predicted class tensor (B,) or scalar
        """
        n_samples = n_samples or self.n_samples
        single    = (x.dim() == 3)
        if single:
            x = x.unsqueeze(0)

        B = x.size(0)
        x = x.to(self.device)
        num_classes = self._num_classes()
        counts = torch.zeros(B, num_classes, device="cpu")

        # Sample in mini-batches
        for start in range(0, n_samples, batch_sz):
            sz   = min(batch_sz, n_samples - start)
            # (B*sz, C, H, W) — repeat each image sz times, add Gaussian noise
            xrep = x.unsqueeze(1).repeat(1, sz, 1, 1, 1).view(B * sz, *x.shape[1:])
            noise = torch.randn_like(xrep) * self.sigma
            noisy = (xrep + noise).clamp(0.0, 1.0)

            logits = self.model(noisy)
            preds  = logits.argmax(dim=1).cpu().view(B, sz)

            for b in range(B):
                for c in preds[b]:
                    counts[b, c.item()] += 1

        preds_final = counts.argmax(dim=1)
        return preds_final.squeeze(0) if single else preds_final

    # ── Certification ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def certify(
        self,
        x:         torch.Tensor,
        n_samples: int | None = None,
        batch_sz:  int        = 64,
    ) -> tuple[int, float]:
        """
        Certify a single image.

        Returns:
            (predicted_class, certified_radius)

        If the prediction confidence is too low (abstain), returns (ABSTAIN, 0.0).

        The certified radius δ is the largest L₂ ball around x within which
        the smoothed classifier is guaranteed to be constant.
        """
        n_samples = n_samples or self.n_samples_cert
        x         = x.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        num_classes = self._num_classes()
        counts = torch.zeros(num_classes, device="cpu")

        for start in range(0, n_samples, batch_sz):
            sz    = min(batch_sz, n_samples - start)
            xrep  = x.repeat(sz, 1, 1, 1)
            noise = torch.randn_like(xrep) * self.sigma
            noisy = (xrep + noise).clamp(0.0, 1.0)
            preds = self.model(noisy).argmax(1).cpu()
            for c in preds:
                counts[c.item()] += 1

        # Top-2 classes
        top2   = counts.topk(2).indices.tolist()
        c_star = top2[0]
        n_star = int(counts[c_star].item())

        # One-sided binomial confidence interval (Clopper-Pearson lower bound)
        p_lower = self._binom_p_lower(n_star, n_samples, self.alpha)

        if p_lower < 0.5:
            return ABSTAIN, 0.0

        # Certified radius: δ = σ · Φ⁻¹(p_lower)
        radius = self.sigma * scipy_norm.ppf(p_lower)
        return c_star, float(radius)

    def certify_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_samples: int | None = None,
    ) -> dict:
        """
        Certify a full dataset.

        Returns metrics dict:
            certified_accuracy  : fraction with radius > 0 and correct pred
            average_radius      : mean certified radius (over certified samples)
            abstain_rate        : fraction that abstain
            results             : list of (true_label, pred, radius)
        """
        results     = []
        n_evaluated = 0

        print(f"\n[Randomized Smoothing] Certifying dataset  (σ={self.sigma}) ...")

        for images, labels in dataloader:
            for i in range(images.size(0)):
                pred, radius = self.certify(images[i])
                results.append({
                    "true":   labels[i].item(),
                    "pred":   pred,
                    "radius": radius,
                })
                n_evaluated += 1
                if max_samples and n_evaluated >= max_samples:
                    break
            if max_samples and n_evaluated >= max_samples:
                break

        total        = len(results)
        certified    = [r for r in results if r["pred"] != ABSTAIN and r["radius"] > 0]
        correct_cert = [r for r in certified if r["pred"] == r["true"]]

        metrics = {
            "n_evaluated":       total,
            "certified_accuracy": len(correct_cert) / total if total else 0.0,
            "average_radius":    float(np.mean([r["radius"] for r in certified])) if certified else 0.0,
            "abstain_rate":      sum(1 for r in results if r["pred"] == ABSTAIN) / total if total else 0.0,
            "results":           results,
        }

        print(f"  CertAcc = {metrics['certified_accuracy']:.3f} | "
              f"ACR = {metrics['average_radius']:.4f} | "
              f"Abstain = {metrics['abstain_rate']:.3f}")
        return metrics

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _num_classes(self) -> int:
        """Infer number of classes from the model's final linear layer."""
        for m in reversed(list(self.model.modules())):
            if isinstance(m, nn.Linear):
                return m.out_features
        raise RuntimeError("Cannot infer num_classes — no Linear layer found.")

    @staticmethod
    def _binom_p_lower(k: int, n: int, alpha: float) -> float:
        """
        Clopper-Pearson lower confidence bound for a binomial proportion.

        P( p >= p_lower ) >= 1 - alpha
        """
        if k == 0:
            return 0.0
        if k == n:
            return (alpha) ** (1.0 / n)   # conservative lower bound when k==n
        from scipy.stats import beta
        return beta.ppf(alpha, k, n - k + 1)
