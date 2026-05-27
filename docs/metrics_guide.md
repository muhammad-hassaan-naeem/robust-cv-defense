# Dual-Metric System — Reference Guide

This document explains both metric systems implemented in this pipeline,
why both exist, and what each metric specifically measures.

---

## Why Two Metric Systems?

Classical metrics and ZKP-informed metrics answer **different questions**:

| Question | System | Metrics |
|----------|--------|---------|
| *How robust is the model under attack?* | Classical | CA, RA, CertAcc, ACR |
| *Can a third party verify the defense claims?* | ZKP-Informed | DPA, IPI, SAS, RSB |

They are **complementary, not alternatives**. A model can have excellent classical
robustness but zero ZKP-verifiability (no audit trail), or excellent ZKP-verifiability
but weak robustness. A deployment-ready safety-critical system needs both.

---

## System 1 — Classical Metrics

### CA — Clean Accuracy
- **Formula:** `correct_clean / total`
- **Purpose:** Establishes the accuracy cost of the defense. Without it, you cannot know
  how much performance the defense layers sacrifice.
- **RobustBench standard:** Yes.

### RA — Robust Accuracy
- **Formula:** `correct_after_attack / total`
- **Variants:** FGSM (ε=8/255), PGD-7 (ε=8/255, 7 steps), PGD-20 (ε=8/255, 20 steps)
- **Gradient masking check:** If `RA(PGD-7) ≈ RA(PGD-20)`, the defense is genuinely
  robust. If `RA(PGD-7) >> RA(PGD-20)`, the defense is masking gradients.
- **RobustBench standard:** Yes.

### CertAcc — Certified Accuracy
- **Formula:** `fraction(pred == true AND delta > 0)`
- **Source:** Randomized Smoothing (Cohen et al., 2019)
- **Guarantee:** Mathematical. No perturbation ‖η‖₂ < δ can change the prediction.
- **ZKP-verifiable:** No — the RS certification procedure itself cannot currently
  be expressed as a ZKP.

### ACR — Average Certified Radius
- **Formula:** `mean(delta)` over certified-correct samples
- **Purpose:** Quantifies *how robust* certified samples are on average.
  Higher ACR = model tolerates larger perturbations with mathematical certainty.
- **ZKP-verifiable:** No.

---

## System 2 — ZKP-Informed Metrics

### ZKP Scope Finding

> **Direct ZKP verification of a neural network's certified robustness radius is
> largely out-of-scope for current ZKP technology.**
>
> Source: Review of existing ZKP-for-AI research (May 2026). Existing ZKP work for
> AI focuses on:
> - Verifying correct execution of the training process (DP-SGD parameters)
> - Verifying provenance/integrity of input images
>
> RSB is included as a forward-looking research target only.

### DPA — DP Adherence ✅ ZKP-Verifiable
- **Formula:** `1.0 if |σ_claimed - σ_actual| ≤ tol AND |C_claimed - C_actual| ≤ tol else 0.0`
- **ZKP proof:** A prover can produce a ZKP that the gradient update at each step
  used exactly the claimed noise multiplier σ and clipping norm C.
- **Practical use:** Regulatory compliance, audit trail for DP privacy budget.

### IPI — Input Provenance Integrity ✅ ZKP-Verifiable
- **Formula:** `fraction(SHA-256(image_at_inference) == committed_reference_hash)`
- **ZKP proof:** A ZKP commitment scheme over the image hash proves the image was
  not modified between the trusted source and the model input.
- **Practical use:** Detects adversarial image substitution attacks at the system
  boundary, independent of model robustness.

### SAS — Sanitization Audit Score ⚠️ Partial
- **Formula:** `fraction(all expected transforms present in transform log)`
- **Auditable:** Yes — deterministic logs are verifiable.
- **ZKP limitation:** The *order* and *presence* of transforms can be logged and
  audited; the *effect* on pixel values cannot currently be ZKP-proved end-to-end.

### RSB — Robustness Soundness Bound ❌ ZKP Out-of-Scope
- **Formula:** Identical to CertAcc — `fraction(pred == true AND delta > 0)`
- **Framing difference:** CertAcc is a robustness claim; RSB is the statement a
  future ZK-RS (Zero-Knowledge Randomized Smoothing) system would need to prove.
- **Research target:** ZK-RS is an open problem. If solved, RSB would become
  the first fully ZKP-verifiable certified robustness metric.

---

## Relationship Between CertAcc and RSB

```
CertAcc (classical) == RSB (ZKP-informed)   [numerically identical]
```

The difference is framing and intent:
- CertAcc: *"The model is provably robust on X% of samples."*
- RSB: *"The statement we want a ZKP to prove is true for X% of samples,
         but we cannot prove it via ZKP today."*

This distinction is critical for honest reporting in safety-critical deployments.

---

## References

- Madry et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks.* ICLR.
- Cohen et al. (2019). *Certified Adversarial Robustness via Randomized Smoothing.* ICML.
- Goodfellow et al. (2014). *Explaining and Harnessing Adversarial Examples.* ICLR.
- Abadi et al. (2016). *Deep Learning with Differential Privacy.* CCS.
- RobustBench: https://robustbench.github.io
