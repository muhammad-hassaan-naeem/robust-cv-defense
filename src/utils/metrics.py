"""
utils/metrics.py
----------------
Dual-metric evaluation system for adversarial defense.

SYSTEM 1 — Classical Metrics (ML robustness standard)
------------------------------------------------------
  CA       Clean Accuracy
  RA       Robust Accuracy  (per attack: FGSM, PGD-7, PGD-20)
  CertAcc  Certified Accuracy  (Randomized Smoothing, Cohen et al.)
  ACR      Average Certified Radius

SYSTEM 2 — ZKP-Informed Metrics (verifiability standard)
---------------------------------------------------------
  DPA  DP Adherence               ZKP-verifiable
  IPI  Input Provenance Integrity  ZKP-verifiable
  SAS  Sanitization Audit Score    Partial (auditable)
  RSB  Robustness Soundness Bound  Out-of-scope for current ZKP tech

ZKP SCOPE NOTE
--------------
Zero-Knowledge Proofs cannot currently verify a neural network's
certified robustness radius end-to-end. ZKP applicability is limited to:
  (1) Verifying DP-SGD training execution (-> DPA)
  (2) Verifying input image provenance    (-> IPI)
RSB is the target statement a future ZK-RS system would prove.
"""

import json
import hashlib
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, field, asdict


ZKP_SCOPE_NOTE = (
    "ZKPs cannot currently verify a neural network's certified robustness radius "
    "end-to-end. ZKP applicability: (1) DP-SGD training execution (DPA), "
    "(2) input image provenance (IPI). RSB (Robustness Soundness Bound) is "
    "out-of-scope for current ZKP technology; it is a ZKP-target metric for future work."
)

EXPECTED_TRANSFORM_ORDER = [
    "adaptive_median_filter",
    "bilateral_denoise",
    "bit_depth_squeeze",
]


# ─────────────────────────────────────────────────────────────────────────────
# Unified DualReport
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DualReport:
    """
    Unified report containing both classical and ZKP-informed metrics.

    Classical metrics answer: "How robust is the model under attack?"
    ZKP metrics answer:       "Can a third party verify the defense claims?"
    """
    model_name: str
    dataset:    str
    eps:        float
    sigma:      float

    # ── Classical metrics ──────────────────────────────────────────────────
    ca:       float = 0.0   # Clean Accuracy
    ra:       dict  = field(default_factory=dict)  # {attack_name: float}
    cert_acc: float = 0.0   # Certified Accuracy
    acr:      float = 0.0   # Average Certified Radius
    classical_abstain_rate: float = 0.0

    # ── ZKP-informed metrics ───────────────────────────────────────────────
    dpa:             float = 0.0   # DP Adherence           ZKP: YES
    ipi:             float = 0.0   # Input Provenance       ZKP: YES
    sas:             float = 0.0   # Sanitization Audit     ZKP: PARTIAL
    rsb:             float = 0.0   # Robustness Bound       ZKP: OUT-OF-SCOPE
    rsb_mean_radius: float = 0.0
    zkp_abstain_rate: float = 0.0

    notes: str = ""

    def print(self):
        w = 68
        sep = "─" * w
        print("\n" + "═" * w)
        print(f"  Dual-Metric Robustness Report — {self.model_name}")
        print(f"  Dataset: {self.dataset}   eps={self.eps:.4f}   sigma={self.sigma:.3f}")
        print("═" * w)

        print(f"\n  {'CLASSICAL METRICS':^{w-2}}")
        print(f"  {'(ML robustness standard — RobustBench compatible)':^{w-2}}")
        print("  " + sep)
        print(f"  {'CA':<8} Clean Accuracy                    {self.ca:>7.4f}")
        for atk, ra_val in self.ra.items():
            print(f"  {'RA':<8} Robust Acc [{atk:<10}]          {ra_val:>7.4f}")
        print(f"  {'CertAcc':<8} Certified Accuracy                {self.cert_acc:>7.4f}")
        print(f"  {'ACR':<8} Avg Certified Radius              {self.acr:>7.4f}")
        print(f"  {'':8} Abstain Rate                      {self.classical_abstain_rate:>7.4f}")

        print(f"\n  {'ZKP-INFORMED METRICS':^{w-2}}")
        print(f"  {'(verifiability standard — deployment audit trail)':^{w-2}}")
        print("  " + sep)
        zkp_rows = [
            ("DPA", "DP Adherence",              self.dpa,             "ZKP ✓ verifiable"),
            ("IPI", "Input Provenance Integrity", self.ipi,             "ZKP ✓ verifiable"),
            ("SAS", "Sanitization Audit Score",   self.sas,             "Partial — auditable"),
            ("RSB", "Robustness Soundness Bound", self.rsb,             "Out-of-scope for ZKP"),
        ]
        for abbr, name, val, zkp_note in zkp_rows:
            print(f"  {abbr:<8} {name:<32} {val:>7.4f}  [{zkp_note}]")
        print(f"  {'':8} RSB Mean Certified Radius        {self.rsb_mean_radius:>7.4f}")
        print(f"  {'':8} Abstain Rate                     {self.zkp_abstain_rate:>7.4f}")

        print(f"\n  !! ZKP SCOPE: {ZKP_SCOPE_NOTE}")
        if self.notes:
            print(f"  NOTE: {self.notes}")
        print("═" * w + "\n")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["zkp_scope_note"] = ZKP_SCOPE_NOTE
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"  [report] Saved -> {path}")

    def summary_dict(self) -> dict:
        """Return flat summary for logging / W&B."""
        out = {
            "classical/ca":       self.ca,
            "classical/cert_acc": self.cert_acc,
            "classical/acr":      self.acr,
            "zkp/dpa":            self.dpa,
            "zkp/ipi":            self.ipi,
            "zkp/sas":            self.sas,
            "zkp/rsb":            self.rsb,
        }
        for atk, val in self.ra.items():
            out[f"classical/ra_{atk.lower().replace('-', '_')}"] = val
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Classical Metric 1 — CA: Clean Accuracy
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def clean_accuracy(model, loader, device, sanitizer=None) -> float:
    """Clean Accuracy (CA): fraction correctly classified on unperturbed inputs."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if sanitizer is not None:
            images = sanitizer(images).to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total   += images.size(0)
    return correct / total if total else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Classical Metric 2 — RA: Robust Accuracy
# ─────────────────────────────────────────────────────────────────────────────
def robust_accuracy(model, loader, attack_fn, device, sanitizer=None) -> float:
    """Robust Accuracy (RA): accuracy after applying attack_fn."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        adv = attack_fn(model, images, labels)  # needs grad — no @no_grad here
        with torch.no_grad():
            if sanitizer is not None:
                adv = sanitizer(adv).to(device)
            correct += (model(adv).argmax(1) == labels).sum().item()
        total += images.size(0)
    return correct / total if total else 0.0


def robust_accuracy_from_corpus(model, corpus, device, sanitizer=None, batch_sz=64) -> dict:
    """RA for every attack in a pre-generated frozen corpus."""
    model.eval()
    results = {}
    for atk_name, (adv_imgs, labels) in corpus.items():
        ds     = torch.utils.data.TensorDataset(adv_imgs, labels)
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_sz)
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                if sanitizer is not None:
                    imgs = sanitizer(imgs).to(device)
                correct += (model(imgs).argmax(1) == lbls).sum().item()
                total   += imgs.size(0)
        results[atk_name] = correct / total if total else 0.0
        print(f"  RA [{atk_name:<12}] = {results[atk_name]:.4f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Classical Metrics 3+4 — CertAcc + ACR  (from RS cert_results)
# ─────────────────────────────────────────────────────────────────────────────
def certified_accuracy_and_acr(cert_results: list) -> tuple:
    """
    Compute CertAcc and ACR from RandomizedSmoother.certify_batch() results.

    Returns:
        (cert_acc, acr, abstain_rate)
    """
    from defense.randomized_smooth import ABSTAIN
    total = len(cert_results)
    if total == 0:
        return 0.0, 0.0, 0.0

    certified_correct = [
        r for r in cert_results
        if r["pred"] != ABSTAIN and r["radius"] > 0 and r["pred"] == r["true"]
    ]
    abstained = [r for r in cert_results if r["pred"] == ABSTAIN]

    cert_acc     = len(certified_correct) / total
    acr          = (sum(r["radius"] for r in certified_correct) / len(certified_correct)
                    if certified_correct else 0.0)
    abstain_rate = len(abstained) / total
    return float(cert_acc), float(acr), float(abstain_rate)


# ─────────────────────────────────────────────────────────────────────────────
# ZKP Metric 1 — DPA: DP Adherence
# ─────────────────────────────────────────────────────────────────────────────
def dp_adherence(
    noise_mult_claimed: float,
    clip_norm_claimed:  float,
    noise_mult_actual:  float,
    clip_norm_actual:   float,
    tolerance:          float = 1e-3,
) -> float:
    """
    DP Adherence (DPA) — ZKP-verifiable.

    Checks whether the actual DP-SGD hyperparameters match the claimed values.
    In a ZKP system the prover supplies a proof; this is the verifier's check.

    Returns 1.0 if both parameters match within tolerance, else 0.0.
    """
    noise_ok = abs(noise_mult_claimed - noise_mult_actual) <= tolerance
    clip_ok  = abs(clip_norm_claimed  - clip_norm_actual)  <= tolerance
    return 1.0 if (noise_ok and clip_ok) else 0.0


def dp_adherence_from_checkpoint(path, claimed_noise_mult, claimed_clip_norm) -> float:
    """Load checkpoint and compute DPA by comparing stored config to claims."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg  = ckpt.get("config", {})
        return dp_adherence(
            noise_mult_claimed = claimed_noise_mult,
            clip_norm_claimed  = claimed_clip_norm,
            noise_mult_actual  = cfg.get("noise_mult",    float("nan")),
            clip_norm_actual   = cfg.get("max_grad_norm", float("nan")),
        )
    except Exception as e:
        print(f"  [DPA] Could not load checkpoint: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# ZKP Metric 2 — IPI: Input Provenance Integrity
# ─────────────────────────────────────────────────────────────────────────────
def image_hash(image: torch.Tensor) -> str:
    """SHA-256 hash of a single image tensor (provenance commitment)."""
    return hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest()


def compute_reference_hashes(images: torch.Tensor) -> list:
    """Compute reference hashes for a batch at the trusted source."""
    return [image_hash(images[i]) for i in range(images.size(0))]


def input_provenance_integrity(images: torch.Tensor, reference_hashes: list) -> float:
    """
    Input Provenance Integrity (IPI) — ZKP-verifiable.

    Simulates a ZKP verifier: re-hashes images at inference time and checks
    against pre-committed reference hashes.

    Returns fraction of images whose hash matches the reference.
    """
    assert images.size(0) == len(reference_hashes)
    matches = sum(
        image_hash(images[i]) == reference_hashes[i]
        for i in range(images.size(0))
    )
    return matches / images.size(0)


# ─────────────────────────────────────────────────────────────────────────────
# ZKP Metric 3 — SAS: Sanitization Audit Score
# ─────────────────────────────────────────────────────────────────────────────
def sanitization_audit_score(transform_log: list, expected_order: list = None) -> float:
    """
    Sanitization Audit Score (SAS) — partially auditable.

    Verifies that the sanitization pipeline applied all expected transforms
    for each input, using a deterministic per-image transform log.

    Returns fraction of inputs with all expected transforms applied.
    """
    if expected_order is None:
        expected_order = EXPECTED_TRANSFORM_ORDER
    if not transform_log:
        return 0.0
    correct = sum(
        all(e in seq for e in expected_order)
        for seq in transform_log
    )
    return correct / len(transform_log)


# ─────────────────────────────────────────────────────────────────────────────
# ZKP Metric 4 — RSB: Robustness Soundness Bound
# ─────────────────────────────────────────────────────────────────────────────
def robustness_soundness_bound(cert_results: list) -> tuple:
    """
    Robustness Soundness Bound (RSB) — ZKP out-of-scope.

    Fraction of samples with certified radius delta > 0 and correct prediction.
    This is numerically identical to CertAcc, but framed as the target statement
    a future ZK-RS proof system would need to prove.

    Per research findings: direct ZKP verification of delta is currently
    out-of-scope. RSB is included as a forward-looking research target only.

    Returns (rsb_score, mean_radius, abstain_rate).
    """
    from defense.randomized_smooth import ABSTAIN
    total = len(cert_results)
    if total == 0:
        return 0.0, 0.0, 0.0

    certified_correct = [
        r for r in cert_results
        if r["pred"] != ABSTAIN and r["radius"] > 0 and r["pred"] == r["true"]
    ]
    abstained    = [r for r in cert_results if r["pred"] == ABSTAIN]
    rsb_score    = len(certified_correct) / total
    mean_radius  = (float(sum(r["radius"] for r in certified_correct) / len(certified_correct))
                    if certified_correct else 0.0)
    abstain_rate = len(abstained) / total
    return rsb_score, mean_radius, abstain_rate


# ─────────────────────────────────────────────────────────────────────────────
# AutoAttack RA — added to DualReport
# ─────────────────────────────────────────────────────────────────────────────
def autoattack_robust_accuracy(model, loader, eps, device, sanitizer=None,
                                steps=100, n_queries=1000) -> float:
    """
    Convenience wrapper — runs AutoAttackEnsemble and returns RA.
    Imported here to keep metrics.py as the single import point for callers.
    """
    from attacks.autoattack import autoattack_accuracy
    return autoattack_accuracy(
        model=model, loader=loader, eps=eps, steps=steps,
        n_queries=n_queries, device=device, sanitizer=sanitizer, verbose=True,
    )
