<div align="center">

# 🛡️ Robust Computer Vision — Adversarial Defense Pipeline

**Muhammad Hassaan Naeem** · Cybersecurity & Machine Learning · May 2026

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Opacus](https://img.shields.io/badge/Opacus-DP--SGD-0668E1)](https://opacus.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-52%20passing-brightgreen)](tests/)

**"Integrating Differential Privacy and Adaptive Denoising for Adversarial Defense in Safety-Critical Systems"**

</div>

---

## Overview

This repository implements a **three-layer adversarial defense pipeline** for Computer Vision models
deployed in safety-critical systems (facial recognition, autonomous vehicles, LiDAR-fused object detection).
It is evaluated using **two complementary metric systems** that together answer different but equally important questions:

| System | Metrics | Question Answered |
|--------|---------|-------------------|
| **Classical** | CA · RA · CertAcc · ACR | *How robust is the model under attack?* |
| **ZKP-Informed** | DPA · IPI · SAS · RSB | *Can a third party cryptographically verify the defense claims?* |

> **Why both?** Classical metrics are the ML community standard (RobustBench-compatible).
> ZKP-informed metrics address deployment-time auditability — critical for regulatory compliance
> in safety-critical systems. They are not substitutes; they answer different questions.

---

## Research Background — ZKP Scope

> **🛡️ Zero-Knowledge Proofs for Robustness are largely out-of-scope for current ZKP technology.**
>
> While ZKPs are an active area of research, direct application to verifying the *certified robustness
> radius* of a neural network prediction was not found to be feasible with current techniques. Existing
> ZKP research for AI focuses on two narrower areas:
>
> 1. **Verifying correct execution of training** — e.g. that DP-SGD noise was applied with the
>    claimed noise multiplier σ and clipping norm C (→ **DPA** metric).
> 2. **Verifying provenance of input images** — e.g. that an image was not tampered with before
>    inference, via cryptographic hash commitments (→ **IPI** metric).
>
> Accordingly, the ZKP-informed metrics in this pipeline measure only what ZKPs *can* actually verify
> today. RSB (Randomized Smoothing certified radius) is reported as a forward-looking research target,
> explicitly flagged as out-of-scope for current ZKP technology.

---

## Dual Metric System

### 📊 Classical Metrics (ML Robustness Standard)

| Metric | Full Name | What it measures |
|--------|-----------|-----------------|
| **CA** | Clean Accuracy | Fraction correctly classified on unperturbed inputs |
| **RA** | Robust Accuracy | Accuracy after FGSM / PGD-7 / PGD-20 attacks |
| **CertAcc** | Certified Accuracy | Fraction with mathematically certified δ > 0 (correct) |
| **ACR** | Average Certified Radius | Mean L₂ radius δ across certified samples |

### 🔐 ZKP-Informed Metrics (Verifiability Standard)

| Metric | ZKP-Verifiable | Full Name | What it measures |
|--------|:--------------:|-----------|-----------------|
| **DPA** | ✅ Yes | DP Adherence | DP-SGD ran with claimed σ and C |
| **IPI** | ✅ Yes | Input Provenance Integrity | No pre-inference image tampering (SHA-256) |
| **SAS** | ⚠️ Partial | Sanitization Audit Score | Correct transforms in correct order |
| **RSB** | ❌ Out-of-scope | Robustness Soundness Bound | Certified fraction (ZKP-target, future work) |

---

## Defense Architecture

```
Raw Input
    │
    ▼  Layer 1 — Input Sanitization (Image Processing)
    │  ├─ Adaptive Median Filter      (removes high-freq adversarial noise)
    │  ├─ Bilateral Denoising         (edge-preserving smoothing)
    │  └─ Bit-Depth Squeezing         (quantizes away fine perturbations)
    │
    ▼  Layer 2 — Adversarially Trained CNN (Machine Learning)
    │  ├─ PGD-7 min-max training      (empirical robustness)
    │  └─ Opacus DP-SGD               (certified gradient sensitivity)
    │
    ▼  Layer 3 — Randomized Smoothing (Differential Privacy)
       ├─ Gaussian noise injection    (σ = 0.25)
       └─ Cohen et al. certification  (L₂ radius δ)
```

**Attack equation:**  `x' = x + ε · sign(∇ₓ L(θ, x, y))`  (FGSM, Goodfellow et al. 2014)

---

## Project Structure

```
robust_cv_defense/
├── src/
│   ├── attacks/
│   │   ├── fgsm.py              # FGSM — single-step white-box attack
│   │   ├── pgd.py               # PGD — iterative L∞ / L2 attack
│   │   └── corpus.py            # Frozen attack corpus generator
│   ├── defense/
│   │   ├── sanitization.py      # Layer 1: IP pre-processing pipeline
│   │   ├── adversarial_train.py # Layer 2: PGD-7 training + Opacus DP-SGD
│   │   └── randomized_smooth.py # Layer 3: Cohen et al. certified smoothing
│   ├── models/
│   │   └── cnn.py               # SmallCNN (MNIST) + RobustCNN (CIFAR-10/GTSRB)
│   ├── evaluation/
│   │   └── evaluator.py         # Dual-metric evaluation engine
│   └── utils/
│       ├── data.py              # CIFAR-10, MNIST, GTSRB dataset loaders
│       └── metrics.py           # Both metric systems + DualReport
├── scripts/
│   ├── run_phase1.py            # Phase I:   generate attack corpus
│   ├── run_phase2.py            # Phase II:  adversarial training
│   ├── run_phase4.py            # Phase IV:  full dual evaluation
│   └── run_all.py               # End-to-end pipeline runner
├── configs/
│   ├── cifar10.yaml             # CIFAR-10 experiment config
│   └── mnist.yaml               # MNIST experiment config
├── tests/
│   └── test_pipeline.py         # 52 unit tests (all passing)
├── docs/
│   └── metrics_guide.md         # Detailed metric explanations
├── .github/
│   └── workflows/ci.yml         # GitHub Actions CI
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/zarka-yousaf/robust-cv-defense.git
cd robust-cv-defense
pip install -e .

# 2. Phase I — generate frozen attack corpus
python scripts/run_phase1.py --config configs/cifar10.yaml

# 3. Phase II — adversarial training (DP-SGD enabled by default)
python scripts/run_phase2.py --config configs/cifar10.yaml --epochs 20

# 4. Phase IV — dual evaluation (classical + ZKP-informed)
python scripts/run_phase4.py --config configs/cifar10.yaml \
                              --checkpoint results/model_best.pt

# 5. End-to-end (all phases)
python scripts/run_all.py --config configs/cifar10.yaml
```

---

## Example Output

```
════════════════════════════════════════════════════════════════════
  Dual-Metric Robustness Report — RobustCNN on CIFAR-10
  eps=0.0314   sigma=0.250   DP-SGD: noise_mult=1.1, clip_norm=1.0
════════════════════════════════════════════════════════════════════

  CLASSICAL METRICS (ML robustness standard)
  ─────────────────────────────────────────────────────────────────
  CA   Clean Accuracy              0.8234
  RA   Robust Acc [FGSM    ]       0.6891
  RA   Robust Acc [PGD-7   ]       0.5912
  RA   Robust Acc [PGD-20  ]       0.5743
  CertAcc  Certified Accuracy      0.4120
  ACR      Avg Certified Radius    0.3847

  ZKP-INFORMED METRICS (verifiability standard)
  ─────────────────────────────────────────────────────────────────
  DPA  DP Adherence          1.0000  [ZKP ✓ verifiable]
  IPI  Input Provenance      1.0000  [ZKP ✓ verifiable]
  SAS  Sanitization Audit    1.0000  [Partial — order auditable]
  RSB  Robustness Bound      0.4120  [Out-of-scope for ZKP]

  !! ZKP SCOPE: RSB cannot currently be proven via ZKP. DPA + IPI
     are the only ZKP-verifiable claims in this pipeline.
════════════════════════════════════════════════════════════════════
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
# 52 tests: models, attacks, sanitization, RS, DPA, IPI, SAS, RSB
```

---

## Citation

```bibtex
@misc{yousaf2026robust,
  title   = {Robust Computer Vision: Integrating Differential Privacy
             and Adaptive Denoising for Adversarial Defense},
  author  = {Naeem, Muhammad Hassaan},
  year    = {2026},
  url     = {https://github.com/muhammad-hassaan-naeem/robust-cv-defense}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Phase V — Research Analysis (New)

Three research additions that upgrade this from a systems project to a paper-ready contribution:

### 1. AutoAttack Evaluation
Community-accepted robustness benchmark (Croce & Hein, 2020). Ensemble of:
- **APGD-CE** — Auto-PGD with cross-entropy loss (adaptive step size)
- **APGD-DLR** — Auto-PGD with difference-of-logits ratio (catches CE saturation)
- **Square** — black-box query-based attack (no gradients, catches gradient masking)

If `RA(AA) ≈ RA(PGD-7)`, the defense is genuinely robust. If `RA(AA) << RA(PGD-7)`, gradient masking is present.

### 2. Accuracy–Robustness Trade-off Curves
σ-sweep over {0.12, 0.25, 0.50, 1.00} showing CA and CertAcc as functions of noise level.
Shows the pipeline offers a **tunable guarantee**, not a single operating point.

### 3. Ablation Study
7-configuration breakdown proving each layer contributes independently:

| Config | Sanitize | Adv Train | RS | Expected Result |
|--------|:--------:|:---------:|:--:|-----------------|
| No Defense | ✗ | ✗ | ✗ | Baseline CA, lowest RA |
| L1 only | ✓ | ✗ | ✗ | Slightly higher RA, no CertAcc |
| L2 only | ✗ | ✓ | ✗ | Highest empirical RA, no CertAcc |
| L3 only | ✗ | ✗ | ✓ | Lower RA, CertAcc appears |
| L1+L2 | ✓ | ✓ | ✗ | Strong RA, no CertAcc |
| L1+L3 | ✓ | ✗ | ✓ | Moderate RA, improved CertAcc |
| **Full (L1+L2+L3)** | ✓ | ✓ | ✓ | **Best on all metrics** |

### Running Phase V

```bash
# Full analysis (sigma sweep + AutoAttack + ablation + figures)
python scripts/run_analysis.py --checkpoint results/model_best.pt

# Fast mode (~5 minutes)
python scripts/run_analysis.py --checkpoint results/model_best.pt --fast

# Skip slow components
python scripts/run_analysis.py --checkpoint results/model_best.pt \
                                --skip_ablation --skip_aa
```

**Output figures** (saved as PDF + PNG for paper submission):
```
figures/
├── fig1_tradeoff_curve.{pdf,png}     # CA vs CertAcc vs σ — Figure 1
├── fig2_radius_distribution.{pdf,png} # δ histograms per σ — Figure 2
├── fig3_ablation.{pdf,png}            # Layer ablation bars — Figure 3
└── fig4_autoattack_vs_pgd.{pdf,png}  # AA vs PGD gap — Figure 4
```
