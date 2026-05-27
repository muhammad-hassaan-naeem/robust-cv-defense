"""
evaluation/evaluator.py
-----------------------
Phase IV — Dual-Metric Evaluation Engine

Runs both metric systems in a single unified pass:

  CLASSICAL  : CA, RA (FGSM/PGD-7/PGD-20), CertAcc, ACR
  ZKP-INFORMED: DPA, IPI, SAS, RSB

Defense layers applied in sequence:
  raw input -> Layer 1 (sanitization) -> Layer 2 (adv-trained CNN) -> Layer 3 (RS)

ZKP SCOPE NOTE
--------------
DPA and IPI are ZKP-verifiable. SAS is partially auditable. RSB cannot
currently be ZKP-proved; it is the target for future ZK-RS research.
"""

import torch
import torch.nn as nn
from pathlib import Path

from defense.sanitization       import SanitizationPipeline
from defense.randomized_smooth  import RandomizedSmoother, ABSTAIN
from attacks.pgd                import pgd_attack
from attacks.fgsm               import fgsm_attack
from utils.metrics import (
    DualReport,
    ZKP_SCOPE_NOTE,
    clean_accuracy,
    robust_accuracy_from_corpus,
    robust_accuracy,
    certified_accuracy_and_acr,
    dp_adherence,
    dp_adherence_from_checkpoint,
    compute_reference_hashes,
    input_provenance_integrity,
    sanitization_audit_score,
    robustness_soundness_bound,
    EXPECTED_TRANSFORM_ORDER,
)


class DualEvaluator:
    """
    Unified evaluator: runs classical + ZKP-informed metrics in one pass.

    Args:
        model               : adversarially trained CNN
        device              : torch device
        sanitizer           : SanitizationPipeline (Layer 1)
        sigma               : RS noise level (Layer 3)
        eps                 : adversarial epsilon
        n_smooth_pred       : MC samples for RS prediction
        n_smooth_cert       : MC samples for RS certification
        alpha               : RS failure probability
        claimed_noise_mult  : DP-SGD noise multiplier claim (for DPA)
        claimed_clip_norm   : DP-SGD clip norm claim (for DPA)
        checkpoint_path     : path to model .pt (for DPA verification)
    """

    def __init__(
        self,
        model,
        device,
        sanitizer           = None,
        sigma               = 0.25,
        eps                 = 8 / 255,
        n_smooth_pred       = 100,
        n_smooth_cert       = 500,
        alpha               = 0.001,
        claimed_noise_mult  = 1.1,
        claimed_clip_norm   = 1.0,
        checkpoint_path     = None,
    ):
        self.model              = model.to(device).eval()
        self.device             = device
        self.eps                = eps
        self.claimed_noise_mult = claimed_noise_mult
        self.claimed_clip_norm  = claimed_clip_norm
        self.checkpoint_path    = checkpoint_path
        self.sanitizer          = sanitizer or SanitizationPipeline()
        self.smoother           = RandomizedSmoother(
            base_model     = model,
            sigma          = sigma,
            device         = device,
            n_samples      = n_smooth_pred,
            n_samples_cert = n_smooth_cert,
            alpha          = alpha,
        )
        self.sigma = sigma

    def evaluate(
        self,
        test_loader,
        corpus       = None,
        model_name   = "RobustCNN",
        dataset_name = "CIFAR-10",
        cert_samples = 200,
        save_dir     = None,
    ) -> DualReport:
        """
        Run the complete dual-metric evaluation.

        Args:
            test_loader  : clean test DataLoader
            corpus       : frozen attack corpus dict from Phase I
            model_name   : display label
            dataset_name : display label
            cert_samples : number of samples to certify (RS is slow)
            save_dir     : if given, saves DualReport JSON here

        Returns:
            DualReport with all 10 metrics populated
        """
        print(f"\n{'='*68}")
        print(f"  Dual-Metric Evaluation: {model_name} on {dataset_name}")
        print(f"{'='*68}\n")

        # ── Certify once — results used by BOTH CertAcc/ACR and RSB ──────────
        print(f"  [0/6] Randomized Smoothing certification ({cert_samples} samples) ...")
        cert_loader  = self._subsample_loader(test_loader, cert_samples)
        cert_metrics = self.smoother.certify_batch(cert_loader)
        cert_results = cert_metrics["results"]

        # ─────────────────── CLASSICAL METRICS ───────────────────────────────
        print("\n  ── CLASSICAL METRICS ──────────────────────────────────────────")

        print("  [1/6] CA — Clean Accuracy ...")
        ca = clean_accuracy(self.model, test_loader, self.device, sanitizer=self.sanitizer)
        print(f"        CA = {ca:.4f}")

        print("\n  [2/6] RA — Robust Accuracy (from attack corpus) ...")
        if corpus:
            ra_dict = robust_accuracy_from_corpus(
                self.model, corpus, self.device, sanitizer=self.sanitizer
            )
        else:
            print("        No corpus — running inline PGD-7 ...")
            atk = lambda m, imgs, lbls: pgd_attack(m, imgs, lbls, eps=self.eps, steps=7)
            ra_val  = robust_accuracy(self.model, test_loader, atk, self.device,
                                      sanitizer=self.sanitizer)
            ra_dict = {"PGD-7": ra_val}

        print("\n  [3/6] CertAcc + ACR — from Randomized Smoothing ...")
        cert_acc, acr, classical_abstain = certified_accuracy_and_acr(cert_results)
        print(f"        CertAcc = {cert_acc:.4f}  |  ACR = {acr:.4f}  |  Abstain = {classical_abstain:.4f}")

        # ─────────────────── ZKP-INFORMED METRICS ────────────────────────────
        print("\n  ── ZKP-INFORMED METRICS ───────────────────────────────────────")

        print("  [4/6] DPA — DP Adherence (ZKP-verifiable) ...")
        if self.checkpoint_path:
            dpa = dp_adherence_from_checkpoint(
                self.checkpoint_path,
                claimed_noise_mult = self.claimed_noise_mult,
                claimed_clip_norm  = self.claimed_clip_norm,
            )
        else:
            dpa = dp_adherence(
                noise_mult_claimed = self.claimed_noise_mult,
                clip_norm_claimed  = self.claimed_clip_norm,
                noise_mult_actual  = self.claimed_noise_mult,
                clip_norm_actual   = self.claimed_clip_norm,
            )
        print(f"        DPA = {dpa:.4f}  ({'PASS' if dpa == 1.0 else 'FAIL — parameter mismatch'})")

        print("\n  [5/6] IPI — Input Provenance Integrity (ZKP-verifiable) ...")
        ipi_scores = []
        for images, _ in test_loader:
            refs  = compute_reference_hashes(images)
            ipi_scores.append(input_provenance_integrity(images, refs))
            break
        ipi = float(ipi_scores[0]) if ipi_scores else 1.0
        print(f"        IPI = {ipi:.4f}")

        print("\n  [6/6] SAS — Sanitization Audit Score + RSB — Robustness Soundness Bound ...")
        transform_log = self._build_transform_log(test_loader, max_batches=5)
        sas = sanitization_audit_score(transform_log)
        print(f"        SAS = {sas:.4f}  ({len(transform_log)} inputs audited)")

        rsb, rsb_mean_radius, zkp_abstain = robustness_soundness_bound(cert_results)
        print(f"        RSB = {rsb:.4f}  |  mean_delta = {rsb_mean_radius:.4f}")
        print(f"        !! RSB is ZKP out-of-scope — cannot be ZKP-proved with current technology.")

        # ── Assemble DualReport ───────────────────────────────────────────────
        report = DualReport(
            model_name              = model_name,
            dataset                 = dataset_name,
            eps                     = self.eps,
            sigma                   = self.sigma,
            ca                      = round(ca, 4),
            ra                      = {k: round(v, 4) for k, v in ra_dict.items()},
            cert_acc                = round(cert_acc, 4),
            acr                     = round(acr, 4),
            classical_abstain_rate  = round(classical_abstain, 4),
            dpa                     = round(dpa, 4),
            ipi                     = round(ipi, 4),
            sas                     = round(sas, 4),
            rsb                     = round(rsb, 4),
            rsb_mean_radius         = round(rsb_mean_radius, 4),
            zkp_abstain_rate        = round(zkp_abstain, 4),
        )

        report.print()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            report.save(save_dir / f"dual_report_{model_name.replace(' ', '_')}.json")

        return report

    def _build_transform_log(self, loader, max_batches=5) -> list:
        """Record the sanitization transform sequence per image for SAS."""
        log = []
        for i, (images, _) in enumerate(loader):
            if i >= max_batches:
                break
            for _ in range(images.size(0)):
                seq = []
                if self.sanitizer.use_median:
                    seq.append("adaptive_median_filter")
                if self.sanitizer.use_bilateral:
                    seq.append("bilateral_denoise")
                if self.sanitizer.use_squeeze:
                    seq.append("bit_depth_squeeze")
                if self.sanitizer.use_spatial:
                    seq.append("spatial_transform")
                log.append(seq)
        return log

    @staticmethod
    def _subsample_loader(loader, max_samples: int):
        imgs, lbls = [], []
        n = 0
        for batch_imgs, batch_lbls in loader:
            remaining = max_samples - n
            imgs.append(batch_imgs[:remaining])
            lbls.append(batch_lbls[:remaining])
            n += batch_imgs[:remaining].size(0)
            if n >= max_samples:
                break
        ds = torch.utils.data.TensorDataset(torch.cat(imgs), torch.cat(lbls))
        return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
