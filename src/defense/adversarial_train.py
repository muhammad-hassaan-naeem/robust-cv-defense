"""
defense/adversarial_train.py
-----------------------------
Layer 2 — Adversarial Training  +  Layer 3 (DP-SGD integration)

Implements the min-max training objective from Madry et al. (2018):

    min_θ  E[(x,y)~D]  max_{η: ‖η‖≤ε}  L(θ, x+η, y)

The outer minimisation uses either standard SGD or Opacus DP-SGD
(Differentially Private SGD), which clips per-sample gradients and
adds calibrated Gaussian noise.  This provides:
  - Empirical robustness  (adversarial training)
  - Certified sensitivity  (DP-SGD gradient clipping + noise)

Usage:
    trainer = AdversarialTrainer(model, device, use_dp=True)
    trainer.train(train_loader, epochs=20)
    trainer.save("results/model_best.pt")
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from attacks.pgd import pgd_attack


# ──────────────────────────────────────────────────────────────────────────────
class AdversarialTrainer:
    """
    Adversarial trainer with optional Differential Privacy (Opacus DP-SGD).

    Args:
        model          : CNN to train
        device         : torch device
        eps            : perturbation budget (L∞, default 8/255)
        alpha          : PGD step size (default 2/255)
        pgd_steps      : inner attack steps (default 7 — PGD-7)
        lr             : learning rate
        weight_decay   : L2 regularisation
        use_dp         : enable Opacus DP-SGD (Layer 3 integration)
        max_grad_norm  : DP clipping norm C
        noise_mult     : DP noise multiplier σ
        delta          : DP delta (target = 1/N for N training samples)
    """

    def __init__(
        self,
        model:          nn.Module,
        device:         torch.device,
        eps:            float = 8 / 255,
        alpha:          float = 2 / 255,
        pgd_steps:      int   = 7,
        lr:             float = 0.01,
        weight_decay:   float = 5e-4,
        use_dp:         bool  = True,
        max_grad_norm:  float = 1.0,
        noise_mult:     float = 1.1,
        delta:          float = 1e-5,
    ):
        self.model         = model.to(device)
        self.device        = device
        self.eps           = eps
        self.alpha         = alpha
        self.pgd_steps     = pgd_steps
        self.lr            = lr
        self.use_dp        = use_dp
        self.max_grad_norm = max_grad_norm
        self.noise_mult    = noise_mult
        self.delta         = delta

        self.loss_fn       = nn.CrossEntropyLoss()
        self.history       = []

        self.optimizer = optim.SGD(
            model.parameters(),
            lr=lr, momentum=0.9, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        # Wrap with Opacus DP engine if requested
        self._dp_engine = None
        if use_dp:
            self._init_dp()

    def _init_dp(self):
        """Attach the Opacus DP engine to the model and optimizer."""
        try:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, _ = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=None,          # attached per-epoch below
                noise_multiplier=self.noise_mult,
                max_grad_norm=self.max_grad_norm,
            )
            self._dp_engine = privacy_engine
            print(f"[DP-SGD] Opacus enabled — noise_mult={self.noise_mult}, clip_norm={self.max_grad_norm}")
        except Exception as e:
            print(f"[DP-SGD] WARNING: Could not initialise Opacus ({e}). Falling back to standard SGD.")
            self.use_dp = False

    def _attach_dp_loader(self, loader):
        """Re-attach data loader to the DP engine (required by Opacus per run)."""
        if self._dp_engine is not None:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, loader = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=loader,
                noise_multiplier=self.noise_mult,
                max_grad_norm=self.max_grad_norm,
            )
            self._dp_engine = privacy_engine
        return loader

    def _train_epoch(self, loader) -> tuple[float, float]:
        """
        One training epoch.

        Returns:
            (avg_loss, clean_accuracy)
        """
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # ── Inner maximisation: generate PGD adversarial examples ────────
            self.model.eval()                       # eval for attack generation
            adv_images = pgd_attack(
                self.model, images, labels,
                eps=self.eps, alpha=self.alpha, steps=self.pgd_steps,
                random_start=True, loss_fn=self.loss_fn,
            )
            self.model.train()                      # back to train mode

            # ── Outer minimisation: update weights on adversarial inputs ─────
            self.optimizer.zero_grad()
            logits = self.model(adv_images)
            loss   = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _evaluate(self, loader, adv: bool = False) -> float:
        """Evaluate clean or adversarial accuracy."""
        self.model.eval()
        correct, total = 0, 0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            if adv:
                images = pgd_attack(
                    self.model, images, labels,
                    eps=self.eps, alpha=self.alpha, steps=self.pgd_steps,
                )
            preds   = self.model(images).argmax(1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)

        return correct / total

    def train(
        self,
        train_loader:  torch.utils.data.DataLoader,
        val_loader:    torch.utils.data.DataLoader | None = None,
        epochs:        int = 20,
        save_dir:      str | Path | None = None,
        verbose:       bool = True,
    ) -> list[dict]:
        """
        Full training loop.

        Args:
            train_loader : training DataLoader
            val_loader   : validation DataLoader (optional — for monitoring)
            epochs       : number of epochs
            save_dir     : if provided, saves best checkpoint here
            verbose      : print per-epoch stats

        Returns:
            training history (list of dicts)
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        best_val_ra = 0.0

        print(f"\n[Phase II] Adversarial Training — {epochs} epochs")
        print(f"  Config: PGD-{self.pgd_steps}, ε={self.eps:.4f}, α={self.alpha:.4f}, DP={self.use_dp}")
        print("-" * 60)

        for epoch in range(1, epochs + 1):
            train_loss, train_ca = self._train_epoch(train_loader)
            self.scheduler.step()

            record = {
                "epoch":      epoch,
                "train_loss": round(train_loss, 4),
                "train_ca":   round(train_ca,   4),
            }

            if val_loader is not None:
                val_ca = self._evaluate(val_loader, adv=False)
                val_ra = self._evaluate(val_loader, adv=True)
                record["val_ca"] = round(val_ca, 4)
                record["val_ra"] = round(val_ra, 4)

                # Save best checkpoint by robust accuracy
                if save_dir and val_ra > best_val_ra:
                    best_val_ra = val_ra
                    self.save(save_dir / "model_best.pt", meta=record)

            if verbose:
                line = (f"  Epoch {epoch:3d}/{epochs} | "
                        f"loss={record['train_loss']:.4f} | "
                        f"CA={record['train_ca']:.3f}")
                if val_loader:
                    line += (f" | val_CA={record.get('val_ca', 0):.3f} "
                             f"| val_RA={record.get('val_ra', 0):.3f}")
                print(line)

            self.history.append(record)

        if save_dir:
            self.save(save_dir / "model_final.pt")
            with open(save_dir / "training_history.json", "w") as f:
                json.dump(self.history, f, indent=2)

        print(f"[Phase II] Training complete. Best val RA = {best_val_ra:.3f}\n")
        return self.history

    def save(self, path: str | Path, meta: dict | None = None):
        """Save model checkpoint."""
        path = Path(path)
        payload = {
            "state_dict": self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "config": {
                "eps": self.eps, "alpha": self.alpha,
                "pgd_steps": self.pgd_steps, "use_dp": self.use_dp,
                "max_grad_norm": self.max_grad_norm, "noise_mult": self.noise_mult,
            },
        }
        if meta:
            payload["meta"] = meta
        torch.save(payload, path)
        print(f"  [ckpt] Saved → {path}")

    @classmethod
    def load_model(cls, path: str | Path, model: nn.Module, device: torch.device) -> nn.Module:
        """Load model weights from a checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
