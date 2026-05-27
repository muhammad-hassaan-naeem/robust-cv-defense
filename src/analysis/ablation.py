"""
analysis/ablation.py
--------------------
Ablation Study — Layer Contributions.

Proves each defense layer contributes independently by running the
evaluation with each combination of layers:

  Config 0: No defense          (baseline — undefended model)
  Config 1: Layer 1 only        (sanitization only)
  Config 2: Layer 2 only        (adversarial training only)
  Config 3: Layer 3 only        (randomized smoothing only)
  Config 4: Layers 1 + 2        (sanitization + adv training)
  Config 5: Layers 1 + 3        (sanitization + RS)
  Config 6: Layers 1 + 2 + 3    (full pipeline — proposed method)

This is the key figure separating a systems paper from a collection
of existing techniques. Without it, reviewers will ask whether each
layer is necessary.

Results are fed directly into analysis/tradeoff_curves.plot_ablation().
"""

import json
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from pathlib import Path

from defense.sanitization       import SanitizationPipeline
from defense.randomized_smooth  import RandomizedSmoother, ABSTAIN
from attacks.pgd                import pgd_attack
from attacks.fgsm               import fgsm_attack
from utils.metrics              import (
    clean_accuracy,
    robust_accuracy,
    certified_accuracy_and_acr,
)
from analysis.tradeoff_curves   import AblationPoint


# ─────────────────────────────────────────────────────────────────────────────
# Ablation configuration definitions
# ─────────────────────────────────────────────────────────────────────────────
ABLATION_CONFIGS = [
    dict(name="No Defense",     use_san=False, use_rs=False, color="#475569"),
    dict(name="L1: Sanitize",   use_san=True,  use_rs=False, color="#f97316"),
    dict(name="L2: Adv Train",  use_san=False, use_rs=False, color="#ef4444",
         note="adv_trained_model"),
    dict(name="L3: RS Only",    use_san=False, use_rs=True,  color="#a855f7"),
    dict(name="L1+L2",          use_san=True,  use_rs=False, color="#22d3ee",
         note="adv_trained_model"),
    dict(name="L1+L3",          use_san=True,  use_rs=True,  color="#60a5fa"),
    dict(name="Full (L1+L2+L3)",use_san=True,  use_rs=True,  color="#34d399",
         note="adv_trained_model"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Ablation runner
# ─────────────────────────────────────────────────────────────────────────────
class AblationRunner:
    """
    Runs the ablation study across all layer configurations.

    Args:
        clean_model    : baseline undefended model
        adv_model      : adversarially trained model (Layer 2)
        test_loader    : test DataLoader
        device         : torch device
        eps            : adversarial epsilon (for RA computation)
        sigma          : RS noise level (for CertAcc computation)
        n_cert         : number of samples to certify (RS is slow)
        n_smooth       : MC samples per certification
        sanitizer_cfg  : SanitizationPipeline kwargs
    """

    def __init__(
        self,
        clean_model,
        adv_model,
        test_loader,
        device,
        eps            = 8 / 255,
        sigma          = 0.25,
        n_cert         = 100,
        n_smooth       = 200,
        sanitizer_cfg  = None,
    ):
        self.clean_model  = clean_model.to(device).eval()
        self.adv_model    = adv_model.to(device).eval()
        self.test_loader  = test_loader
        self.device       = device
        self.eps          = eps
        self.sigma        = sigma
        self.n_cert       = n_cert
        self.n_smooth     = n_smooth

        cfg = sanitizer_cfg or {}
        self.sanitizer = SanitizationPipeline(
            use_median    = cfg.get("use_median",    True),
            use_bilateral = cfg.get("use_bilateral", True),
            use_squeeze   = cfg.get("use_squeeze",   True),
            use_spatial   = False,
        )

    def run(self, save_path: Path = None) -> list:
        """
        Run ablation across all 7 configurations.

        Returns:
            List of AblationPoint, one per configuration
        """
        results = []
        pgd7_fn = lambda m, imgs, lbls: pgd_attack(
            m, imgs, lbls, eps=self.eps, alpha=2/255, steps=7
        )

        print(f"\n{'='*60}")
        print(f"  Ablation Study — {len(ABLATION_CONFIGS)} configurations")
        print(f"  σ={self.sigma}  ε={self.eps:.4f}  n_cert={self.n_cert}")
        print(f"{'='*60}\n")

        for i, cfg in enumerate(ABLATION_CONFIGS):
            name     = cfg["name"]
            use_san  = cfg["use_san"]
            use_rs   = cfg["use_rs"]
            use_adv  = "adv_trained_model" in cfg.get("note", "")

            print(f"  [{i+1}/{len(ABLATION_CONFIGS)}] {name}")
            print(f"        sanitize={use_san}  adv_train={use_adv}  RS={use_rs}")

            model     = self.adv_model if use_adv else self.clean_model
            sanitizer = self.sanitizer if use_san else None

            # CA
            ca = clean_accuracy(model, self.test_loader, self.device, sanitizer=sanitizer)
            print(f"        CA = {ca:.4f}")

            # RA (PGD-7)
            ra_pgd7 = robust_accuracy(model, self.test_loader, pgd7_fn,
                                      self.device, sanitizer=sanitizer)
            print(f"        RA (PGD-7) = {ra_pgd7:.4f}")

            # CertAcc + ACR via RS
            if use_rs:
                smoother     = RandomizedSmoother(
                    model, sigma=self.sigma, device=self.device,
                    n_samples=self.n_smooth, n_samples_cert=self.n_smooth,
                )
                cert_loader  = self._subsample(self.n_cert)
                cert_metrics = smoother.certify_batch(cert_loader)
                cert_acc, acr, _ = certified_accuracy_and_acr(cert_metrics["results"])
            else:
                cert_acc, acr = 0.0, 0.0
            print(f"        CertAcc = {cert_acc:.4f}  ACR = {acr:.4f}")

            results.append(AblationPoint(
                label    = name,
                ca       = round(ca, 4),
                ra_pgd7  = round(ra_pgd7, 4),
                cert_acc = round(cert_acc, 4),
                acr      = round(acr, 4),
                color    = cfg["color"],
            ))
            print()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            data = [{"label": p.label, "ca": p.ca, "ra_pgd7": p.ra_pgd7,
                     "cert_acc": p.cert_acc, "acr": p.acr, "color": p.color}
                    for p in results]
            with open(save_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  [ablation] Results saved -> {save_path}")

        return results

    def _subsample(self, n: int):
        imgs, lbls = [], []
        total = 0
        for batch_imgs, batch_lbls in self.test_loader:
            rem = n - total
            imgs.append(batch_imgs[:rem])
            lbls.append(batch_lbls[:rem])
            total += batch_imgs[:rem].size(0)
            if total >= n:
                break
        ds = torch.utils.data.TensorDataset(torch.cat(imgs), torch.cat(lbls))
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)


def load_ablation_results(path: Path) -> list:
    """Load saved ablation results from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [AblationPoint(**d) for d in data]
