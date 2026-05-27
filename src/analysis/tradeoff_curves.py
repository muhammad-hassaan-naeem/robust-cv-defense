"""
analysis/tradeoff_curves.py
----------------------------
Accuracy–Robustness Trade-off Curves.

The central empirical figure in every Randomized Smoothing paper.

Plots CA and CertAcc as σ (noise level) varies across a grid:
    σ ∈ {0.12, 0.25, 0.50, 1.00}

Shows that σ controls a tunable guarantee:
  - Low σ  → high clean accuracy, small certified radius
  - High σ → lower clean accuracy, large certified radius

The curve is what separates a single operating point from a
systematic finding. Without it, certification results look like
one data point rather than a contribution.

Also plots:
  - RA vs σ (robust accuracy under PGD-7 and FGSM)
  - Certified radius distribution histogram at each σ
  - Ablation comparison (see analysis/ablation.py for ablation runner)

All figures are saved as publication-quality PDFs + PNGs.
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SigmaPoint:
    """Results at one σ operating point."""
    sigma:       float
    ca:          float          # Clean Accuracy
    ra_fgsm:     float = 0.0   # Robust Accuracy — FGSM
    ra_pgd7:     float = 0.0   # Robust Accuracy — PGD-7
    ra_aa:       float = 0.0   # Robust Accuracy — AutoAttack
    cert_acc:    float = 0.0   # Certified Accuracy
    acr:         float = 0.0   # Average Certified Radius
    abstain:     float = 0.0
    radii:       list  = field(default_factory=list)  # per-sample certified radii


@dataclass
class AblationPoint:
    """Results for one ablation configuration."""
    label:    str
    ca:       float
    ra_pgd7:  float
    cert_acc: float
    acr:      float
    color:    str = "#60a5fa"


# ─────────────────────────────────────────────────────────────────────────────
# Plot style
# ─────────────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor":  "#0a0f1e",
    "axes.facecolor":    "#0f172a",
    "axes.edgecolor":    "#1e293b",
    "axes.labelcolor":   "#cbd5e1",
    "axes.titlecolor":   "#f1f5f9",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "grid.color":        "#1e293b",
    "grid.linewidth":    0.8,
    "text.color":        "#cbd5e1",
    "legend.facecolor":  "#0f172a",
    "legend.edgecolor":  "#1e293b",
    "legend.labelcolor": "#cbd5e1",
    "font.family":       "DejaVu Sans",
    "font.size":         10,
}

CA_COLOR       = "#60a5fa"   # blue
CERTACC_COLOR  = "#a855f7"   # purple
ACR_COLOR      = "#8b5cf6"   # violet
RA_FGSM_COLOR  = "#f97316"   # orange
RA_PGD_COLOR   = "#ef4444"   # red
RA_AA_COLOR    = "#dc2626"   # deep red
SIGMA_COLORS   = ["#22d3ee", "#60a5fa", "#a855f7", "#f97316"]


def _apply_style():
    plt.rcParams.update(STYLE)


def _save(fig, path: Path, stem: str):
    """Save figure as both PDF and PNG."""
    path.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fp = path / f"{stem}.{ext}"
        fig.savefig(fp, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  [figure] Saved -> {fp}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — CA vs CertAcc vs σ  (THE main trade-off curve)
# ─────────────────────────────────────────────────────────────────────────────
def plot_tradeoff_curve(
    sigma_points: list,
    save_dir:     Path,
    title:        str = "Accuracy–Robustness Trade-off",
    show_aa:      bool = True,
):
    """
    Plot CA, CertAcc, and RA as functions of σ.

    This is Figure 1 in the paper — the primary empirical contribution
    of the Randomized Smoothing analysis.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(STYLE["figure.facecolor"])

    sigmas    = [p.sigma    for p in sigma_points]
    cas       = [p.ca       for p in sigma_points]
    cert_accs = [p.cert_acc for p in sigma_points]
    acrs      = [p.acr      for p in sigma_points]
    ra_fgsms  = [p.ra_fgsm  for p in sigma_points]
    ra_pgd7s  = [p.ra_pgd7  for p in sigma_points]
    ra_aas    = [p.ra_aa    for p in sigma_points]

    # ── Left: Accuracy vs σ ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(sigmas, cas,       "o-", color=CA_COLOR,      lw=2.2, ms=7, label="CA (Clean)")
    ax.plot(sigmas, cert_accs, "s-", color=CERTACC_COLOR, lw=2.2, ms=7, label="CertAcc (Certified)")
    ax.plot(sigmas, ra_fgsms,  "^-", color=RA_FGSM_COLOR, lw=1.8, ms=6, label="RA (FGSM)")
    ax.plot(sigmas, ra_pgd7s,  "v-", color=RA_PGD_COLOR,  lw=1.8, ms=6, label="RA (PGD-7)")
    if show_aa and any(p.ra_aa > 0 for p in sigma_points):
        ax.plot(sigmas, ra_aas, "D-", color=RA_AA_COLOR, lw=1.8, ms=6, label="RA (AutoAttack)")

    ax.set_xlabel("Noise level σ", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy vs σ", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(sigmas)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.4)

    # Shade the CA–CertAcc gap (robustness cost zone)
    ax.fill_between(sigmas, cert_accs, cas, alpha=0.1, color=CERTACC_COLOR,
                    label="_nolegend_")
    ax.annotate("accuracy–\ncertification gap", xy=(sigmas[1], (cas[1] + cert_accs[1]) / 2),
                xytext=(sigmas[1] + 0.05, (cas[1] + cert_accs[1]) / 2 + 0.1),
                fontsize=8, color=CERTACC_COLOR, alpha=0.8,
                arrowprops=dict(arrowstyle="->", color=CERTACC_COLOR, lw=0.8))

    # ── Right: ACR vs σ ──────────────────────────────────────────────────────
    ax2 = axes[1]
    bars = ax2.bar([str(s) for s in sigmas], acrs,
                   color=SIGMA_COLORS[:len(sigmas)], alpha=0.85, width=0.5, zorder=3)
    ax2.set_xlabel("Noise level σ", fontsize=11)
    ax2.set_ylabel("Average Certified Radius (δ̄)", fontsize=11)
    ax2.set_title("ACR vs σ", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.4, axis="y")

    for bar, val in zip(bars, acrs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                 color="#f1f5f9", fontweight="bold")

    fig.suptitle(f"{title}\n(ε = 8/255, L∞)", fontsize=13, fontweight="bold",
                 color="#f1f5f9", y=1.01)
    plt.tight_layout()
    _save(fig, save_dir, "fig1_tradeoff_curve")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Certified radius distribution histograms
# ─────────────────────────────────────────────────────────────────────────────
def plot_radius_distribution(
    sigma_points: list,
    save_dir:     Path,
    title:        str = "Certified Radius Distribution",
):
    """
    Plot histograms of per-sample certified radii δ for each σ.

    Shows whether samples cluster near δ=0 (weak certification) or
    spread across larger radii (strong certification). This is standard
    in RS papers (Cohen et al., 2019; Salman et al., 2019).
    """
    _apply_style()
    n = len(sigma_points)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor(STYLE["figure.facecolor"])

    for i, (sp, color) in enumerate(zip(sigma_points, SIGMA_COLORS)):
        ax = axes[i]
        radii = [r for r in sp.radii if r > 0]  # exclude abstain/zero

        if radii:
            ax.hist(radii, bins=25, color=color, alpha=0.8, edgecolor="#0f172a", zorder=3)
            ax.axvline(np.mean(radii), color="#fbbf24", lw=1.5, linestyle="--",
                       label=f"ACR = {np.mean(radii):.3f}")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No certified\nsamples", ha="center", va="center",
                    transform=ax.transAxes, color="#64748b")

        ax.set_title(f"σ = {sp.sigma}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Certified radius δ", fontsize=10)
        if i == 0:
            ax.set_ylabel("Sample count", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # Annotation: fraction certified
        cert_frac = sum(1 for r in sp.radii if r > 0) / len(sp.radii) if sp.radii else 0
        ax.text(0.97, 0.97, f"CertAcc = {cert_frac:.2f}", ha="right", va="top",
                transform=ax.transAxes, fontsize=8, color="#94a3b8",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", alpha=0.8))

    fig.suptitle(title, fontsize=12, fontweight="bold", color="#f1f5f9")
    plt.tight_layout()
    _save(fig, save_dir, "fig2_radius_distribution")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Ablation comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_ablation(
    ablation_points: list,
    save_dir:        Path,
    title:           str = "Ablation Study — Layer Contributions",
):
    """
    Bar chart comparing defense configurations.

    Shows each layer's individual contribution and the combined effect.
    This is the key figure proving the three-layer design is justified.
    """
    _apply_style()
    metrics = ["CA", "RA (PGD-7)", "CertAcc", "ACR"]
    n_cfg   = len(ablation_points)
    x       = np.arange(len(metrics))
    width   = 0.8 / n_cfg

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(STYLE["figure.facecolor"])

    ABLATION_COLORS = [
        "#475569",   # gray    — no defense
        "#f97316",   # orange  — Layer 1 only
        "#ef4444",   # red     — Layer 2 only
        "#a855f7",   # purple  — Layer 3 only
        "#22d3ee",   # cyan    — L1 + L2
        "#60a5fa",   # blue    — L1 + L3
        "#34d399",   # green   — full (L1+L2+L3)
    ]

    for i, pt in enumerate(ablation_points):
        vals   = [pt.ca, pt.ra_pgd7, pt.cert_acc, pt.acr]
        offset = (i - n_cfg / 2 + 0.5) * width
        color  = ABLATION_COLORS[i % len(ABLATION_COLORS)]
        bars   = ax.bar(x + offset, vals, width * 0.9, label=pt.label,
                        color=color, alpha=0.85, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, color="#f1f5f9", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # Highlight full-defense bar
    ax.axhline(ablation_points[-1].ca, color=CA_COLOR, lw=0.8,
               linestyle=":", alpha=0.5, label="_nolegend_")

    plt.tight_layout()
    _save(fig, save_dir, "fig3_ablation")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — AutoAttack vs PGD comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_autoattack_comparison(
    sigma_points: list,
    save_dir:     Path,
    title:        str = "AutoAttack vs PGD — Gradient Masking Check",
):
    """
    Side-by-side RA comparison: PGD-20 vs AutoAttack.

    If AA-RA << PGD-RA, gradient masking is present.
    If AA-RA ≈ PGD-RA, the defense is genuinely robust.
    """
    if not any(p.ra_aa > 0 for p in sigma_points):
        print("  [tradeoff] No AutoAttack results — skipping Fig 4.")
        return

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(STYLE["figure.facecolor"])

    sigmas   = [str(p.sigma) for p in sigma_points]
    ra_pgd7s = [p.ra_pgd7   for p in sigma_points]
    ra_aas   = [p.ra_aa     for p in sigma_points]

    x     = np.arange(len(sigmas))
    width = 0.35

    b1 = ax.bar(x - width/2, ra_pgd7s, width, label="RA (PGD-7)",    color=RA_PGD_COLOR, alpha=0.85, zorder=3)
    b2 = ax.bar(x + width/2, ra_aas,   width, label="RA (AutoAttack)", color=RA_AA_COLOR,  alpha=0.85, zorder=3)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9,
                    color="#f1f5f9", fontweight="bold")

    # Gap annotations
    for i, (p7, aa) in enumerate(zip(ra_pgd7s, ra_aas)):
        gap = p7 - aa
        if gap > 0.01:
            ax.annotate(f"Δ={gap:.3f}", xy=(i, max(p7, aa) + 0.05),
                        ha="center", fontsize=8, color="#fbbf24",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels([f"σ={s}" for s in sigmas], fontsize=10)
    ax.set_ylabel("Robust Accuracy", fontsize=11)
    ax.set_title(title + "\n(small gap → no gradient masking)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save(fig, save_dir, "fig4_autoattack_vs_pgd")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Save/load sigma sweep results
# ─────────────────────────────────────────────────────────────────────────────
def save_sigma_sweep(points: list, path: Path):
    data = []
    for p in points:
        d = {k: v for k, v in p.__dict__.items() if k != "radii"}
        d["radii"] = p.radii
        data.append(d)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [sweep] Saved {len(points)} sigma points -> {path}")


def load_sigma_sweep(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)
    return [SigmaPoint(**d) for d in data]
