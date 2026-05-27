#!/usr/bin/env python3
"""
scripts/run_analysis.py
-----------------------
Phase V — Full Research Analysis

Runs all three research additions in one script:

  1. AutoAttack evaluation   (reliable robustness benchmark)
  2. σ-sweep                 (accuracy–robustness trade-off curves)
  3. Ablation study          (layer contribution analysis)

Then generates all four publication figures:
  fig1_tradeoff_curve.{pdf,png}
  fig2_radius_distribution.{pdf,png}
  fig3_ablation.{pdf,png}
  fig4_autoattack_vs_pgd.{pdf,png}

Usage:
    python scripts/run_analysis.py --checkpoint results/model_best.pt
    python scripts/run_analysis.py --checkpoint results/model_best.pt \\
                                    --fast          # fewer samples, faster run
    python scripts/run_analysis.py --checkpoint results/model_best.pt \\
                                    --skip_aa       # skip AutoAttack (slow)
    python scripts/run_analysis.py --checkpoint results/model_best.pt \\
                                    --skip_ablation # skip ablation (very slow)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import argparse, json, torch
from pathlib import Path

from models.cnn                  import get_model
from utils.data                  import get_dataset
from defense.sanitization        import SanitizationPipeline
from defense.adversarial_train   import AdversarialTrainer
from defense.randomized_smooth   import RandomizedSmoother, ABSTAIN
from attacks.fgsm                import fgsm_attack
from attacks.pgd                 import pgd_attack
from attacks.autoattack          import autoattack_accuracy
from utils.metrics               import (
    clean_accuracy, robust_accuracy, certified_accuracy_and_acr,
)
from analysis.tradeoff_curves    import (
    SigmaPoint, plot_tradeoff_curve, plot_radius_distribution,
    plot_autoattack_comparison, save_sigma_sweep,
)
from analysis.ablation           import AblationRunner, load_ablation_results
from analysis.tradeoff_curves    import plot_ablation


def evaluate_at_sigma(model, loader, sanitizer, device, sigma,
                      eps, n_cert, n_smooth, aa_steps, aa_queries,
                      run_aa=True):
    """Evaluate all metrics at one σ operating point."""
    print(f"\n  ── σ = {sigma} ──────────────────────────────────────────────────")

    # CA
    ca = clean_accuracy(model, loader, device, sanitizer=sanitizer)
    print(f"    CA       = {ca:.4f}")

    # RA — FGSM
    fgsm_fn = lambda m, imgs, lbls: fgsm_attack(m, imgs, lbls, eps=eps)
    ra_fgsm = robust_accuracy(model, loader, fgsm_fn, device, sanitizer=sanitizer)
    print(f"    RA-FGSM  = {ra_fgsm:.4f}")

    # RA — PGD-7
    pgd7_fn = lambda m, imgs, lbls: pgd_attack(m, imgs, lbls, eps=eps, steps=7)
    ra_pgd7 = robust_accuracy(model, loader, pgd7_fn, device, sanitizer=sanitizer)
    print(f"    RA-PGD7  = {ra_pgd7:.4f}")

    # AutoAttack
    ra_aa = 0.0
    if run_aa:
        print(f"    Running AutoAttack (steps={aa_steps}, queries={aa_queries}) ...")
        # Use small subset for AA (it's slow)
        aa_imgs, aa_lbls = [], []
        n = 0
        for imgs, lbls in loader:
            rem = min(50, n_cert) - n
            if rem <= 0: break
            aa_imgs.append(imgs[:rem]); aa_lbls.append(lbls[:rem])
            n += imgs[:rem].size(0)
        aa_imgs = torch.cat(aa_imgs).to(device)
        aa_lbls = torch.cat(aa_lbls).to(device)
        ds      = torch.utils.data.TensorDataset(aa_imgs, aa_lbls)
        aa_ld   = torch.utils.data.DataLoader(ds, batch_size=32)
        ra_aa   = autoattack_accuracy(model, aa_ld, eps=eps, steps=aa_steps,
                                       n_queries=aa_queries, device=device,
                                       sanitizer=sanitizer, verbose=True)
        print(f"    RA-AA    = {ra_aa:.4f}")

    # CertAcc + ACR + radius distribution
    smoother    = RandomizedSmoother(model, sigma=sigma, device=device,
                                     n_samples=50, n_samples_cert=n_smooth)
    cert_imgs, cert_lbls = [], []
    n = 0
    for imgs, lbls in loader:
        rem = n_cert - n
        if rem <= 0: break
        cert_imgs.append(imgs[:rem]); cert_lbls.append(lbls[:rem]); n += imgs[:rem].size(0)
    cert_ds  = torch.utils.data.TensorDataset(torch.cat(cert_imgs), torch.cat(cert_lbls))
    cert_ld  = torch.utils.data.DataLoader(cert_ds, batch_size=1)
    cert_res = smoother.certify_batch(cert_ld)
    cert_acc, acr, abstain = certified_accuracy_and_acr(cert_res["results"])
    radii = [r["radius"] for r in cert_res["results"]]
    print(f"    CertAcc  = {cert_acc:.4f}  ACR = {acr:.4f}  Abstain = {abstain:.4f}")

    return SigmaPoint(
        sigma=sigma, ca=round(ca,4), ra_fgsm=round(ra_fgsm,4),
        ra_pgd7=round(ra_pgd7,4), ra_aa=round(ra_aa,4),
        cert_acc=round(cert_acc,4), acr=round(acr,4),
        abstain=round(abstain,4), radii=radii,
    )


def main():
    p = argparse.ArgumentParser(description="Phase V — Research Analysis")
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--dataset",       default="mnist", choices=["cifar10","mnist"])
    p.add_argument("--eps",           type=float, default=8/255)
    p.add_argument("--sigmas",        nargs="+",  type=float, default=[0.12, 0.25, 0.50, 1.00])
    p.add_argument("--n_cert",        type=int,   default=100,
                   help="Samples to certify per sigma point")
    p.add_argument("--n_smooth",      type=int,   default=200,
                   help="MC samples per RS certification")
    p.add_argument("--aa_steps",      type=int,   default=50,
                   help="AutoAttack APGD steps (100=standard, 50=fast)")
    p.add_argument("--aa_queries",    type=int,   default=500,
                   help="AutoAttack Square queries (1000=standard, 500=fast)")
    p.add_argument("--skip_aa",      action="store_true", help="Skip AutoAttack")
    p.add_argument("--skip_ablation",action="store_true", help="Skip ablation study")
    p.add_argument("--fast",         action="store_true",
                   help="Fast mode: fewer samples, fewer steps")
    p.add_argument("--data_dir",      default="./data")
    p.add_argument("--results_dir",   default="./results")
    p.add_argument("--figures_dir",   default="./figures")
    args = p.parse_args()

    if args.fast:
        args.n_cert   = 30
        args.n_smooth = 100
        args.aa_steps = 20
        args.aa_queries = 100
        print("[fast mode] n_cert=30, n_smooth=100, aa_steps=20, aa_queries=100")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    figures_dir = Path(args.figures_dir)
    results_dir = Path(args.results_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model and data ───────────────────────────────────────────────────
    _, _, test_loader = get_dataset(args.dataset, args.data_dir, batch_size=64)
    arch   = "robust" if args.dataset == "cifar10" else "small"
    model  = get_model(arch)
    model  = AdversarialTrainer.load_model(args.checkpoint, model, device)
    sanitizer = SanitizationPipeline(use_median=True, use_bilateral=True,
                                      use_squeeze=True, use_spatial=False)

    # ── 1. σ-SWEEP + AutoAttack ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase V-A: σ-sweep over {args.sigmas}")
    print(f"{'='*60}")

    sigma_points = []
    for sigma in args.sigmas:
        pt = evaluate_at_sigma(
            model       = model,
            loader      = test_loader,
            sanitizer   = sanitizer,
            device      = device,
            sigma       = sigma,
            eps         = args.eps,
            n_cert      = args.n_cert,
            n_smooth    = args.n_smooth,
            aa_steps    = args.aa_steps,
            aa_queries  = args.aa_queries,
            run_aa      = not args.skip_aa,
        )
        sigma_points.append(pt)

    sweep_path = results_dir / "sigma_sweep.json"
    save_sigma_sweep(sigma_points, sweep_path)

    # ── 2. Generate figures 1, 2, 4 ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Phase V-B: Generating figures ...")
    print(f"{'='*60}")

    title_base = f"RobustCNN · {args.dataset.upper()} · ε={args.eps:.4f}"

    plot_tradeoff_curve(
        sigma_points, figures_dir,
        title   = title_base,
        show_aa = not args.skip_aa,
    )
    plot_radius_distribution(sigma_points, figures_dir, title=title_base)
    if not args.skip_aa:
        plot_autoattack_comparison(sigma_points, figures_dir, title=title_base)

    # ── 3. Ablation study ─────────────────────────────────────────────────────
    if not args.skip_ablation:
        print(f"\n{'='*60}")
        print("  Phase V-C: Ablation study ...")
        print(f"{'='*60}")

        # Load clean baseline model for ablation
        clean_model = get_model(arch)
        baseline_pt = results_dir / "corpus" / "model_baseline.pt"
        if baseline_pt.exists():
            clean_model.load_state_dict(
                torch.load(baseline_pt, map_location=device, weights_only=True)
            )
        else:
            print(f"  [ablation] No baseline model at {baseline_pt} — using random weights.")
        clean_model = clean_model.to(device).eval()

        runner = AblationRunner(
            clean_model   = clean_model,
            adv_model     = model,
            test_loader   = test_loader,
            device        = device,
            eps           = args.eps,
            sigma         = args.sigmas[1] if len(args.sigmas) > 1 else 0.25,
            n_cert        = args.n_cert,
            n_smooth      = args.n_smooth,
        )
        ablation_path   = results_dir / "ablation_results.json"
        ablation_points = runner.run(save_path=ablation_path)
        plot_ablation(ablation_points, figures_dir, title=f"Ablation — {args.dataset.upper()}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Phase V COMPLETE")
    print(f"  Figures -> {figures_dir}/")
    print(f"  fig1_tradeoff_curve.{{pdf,png}}")
    print(f"  fig2_radius_distribution.{{pdf,png}}")
    if not args.skip_aa:
        print(f"  fig4_autoattack_vs_pgd.{{pdf,png}}")
    if not args.skip_ablation:
        print(f"  fig3_ablation.{{pdf,png}}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
