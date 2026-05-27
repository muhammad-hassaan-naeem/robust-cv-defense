#!/usr/bin/env python3
"""
End-to-end pipeline — all phases in sequence.

  Phase I   : train clean baseline + generate attack corpus
  Phase II  : adversarial training with DP-SGD
  Phase IV  : dual-metric evaluation (classical + ZKP-informed)
  Phase V   : research analysis (AutoAttack, σ-sweep, ablation, figures)

Usage:
    python scripts/run_all.py --dataset mnist
    python scripts/run_all.py --dataset cifar10 --epochs 20
    python scripts/run_all.py --dataset mnist --fast --skip_ablation
"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
import argparse, subprocess
from pathlib import Path


def run(cmd: list):
    print(f"\n{'='*60}\n  >> {' '.join(str(c) for c in cmd)}\n{'='*60}")
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       default="mnist",   choices=["cifar10","mnist"])
    p.add_argument("--epochs",        type=int,  default=10)
    p.add_argument("--p1_epochs",     type=int,  default=3)
    p.add_argument("--cert_samples",  type=int,  default=100)
    p.add_argument("--data_dir",      default="./data")
    p.add_argument("--results_dir",   default="./results")
    p.add_argument("--figures_dir",   default="./figures")
    p.add_argument("--no_dp",        action="store_true")
    p.add_argument("--fast",         action="store_true",
                   help="Fast mode for Phase V (fewer samples)")
    p.add_argument("--skip_ablation",action="store_true")
    p.add_argument("--skip_aa",      action="store_true")
    p.add_argument("--skip_phase5",  action="store_true",
                   help="Skip Phase V (analysis + figures)")
    args = p.parse_args()

    scripts = Path(__file__).parent
    corpus  = f"{args.results_dir}/corpus"
    ckpt    = f"{args.results_dir}/model_best.pt"

    # Phase I
    run(["python", scripts/"run_phase1.py",
         "--dataset", args.dataset, "--epochs", args.p1_epochs,
         "--data_dir", args.data_dir, "--save_dir", corpus])

    # Phase II
    cmd2 = ["python", scripts/"run_phase2.py",
            "--dataset", args.dataset, "--epochs", args.epochs,
            "--data_dir", args.data_dir, "--save_dir", args.results_dir]
    if args.no_dp: cmd2.append("--no_dp")
    run(cmd2)

    # Phase IV
    run(["python", scripts/"run_phase4.py",
         "--checkpoint", ckpt, "--dataset", args.dataset,
         "--corpus_dir", corpus, "--cert_samples", args.cert_samples,
         "--data_dir", args.data_dir, "--save_dir", args.results_dir])

    # Phase V
    if not args.skip_phase5:
        cmd5 = ["python", scripts/"run_analysis.py",
                "--checkpoint", ckpt, "--dataset", args.dataset,
                "--data_dir", args.data_dir,
                "--results_dir", args.results_dir,
                "--figures_dir", args.figures_dir]
        if args.fast:          cmd5.append("--fast")
        if args.skip_ablation: cmd5.append("--skip_ablation")
        if args.skip_aa:       cmd5.append("--skip_aa")
        run(cmd5)

    print(f"\n{'='*60}")
    print("  ALL PHASES COMPLETE")
    print(f"  Results  -> {args.results_dir}/")
    print(f"  Figures  -> {args.figures_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
