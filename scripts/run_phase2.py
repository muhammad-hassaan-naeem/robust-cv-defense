#!/usr/bin/env python3
"""
Phase II+III — Adversarial Training with DP-SGD.

Layer 2: PGD-7 min-max adversarial training.
Layer 3: Opacus DP-SGD (certified gradient sensitivity).

Usage:
    python scripts/run_phase2.py --dataset cifar10 --epochs 20
    python scripts/run_phase2.py --dataset mnist   --epochs 10 --no_dp
"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
import argparse
import torch
from models.cnn              import get_model
from utils.data              import get_dataset
from defense.adversarial_train import AdversarialTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",       default="cifar10", choices=["cifar10","mnist"])
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--eps",           type=float, default=8/255)
    p.add_argument("--alpha",         type=float, default=2/255)
    p.add_argument("--pgd_steps",     type=int,   default=7)
    p.add_argument("--lr",            type=float, default=0.01)
    p.add_argument("--noise_mult",    type=float, default=1.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--no_dp",        action="store_true")
    p.add_argument("--data_dir",      default="./data")
    p.add_argument("--save_dir",      default="./results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  DP-SGD: {not args.no_dp}")

    train_loader, val_loader, _ = get_dataset(args.dataset, args.data_dir, args.batch_size)
    arch    = "robust" if args.dataset == "cifar10" else "small"
    model   = get_model(arch)
    trainer = AdversarialTrainer(
        model=model, device=device, eps=args.eps, alpha=args.alpha,
        pgd_steps=args.pgd_steps, lr=args.lr, use_dp=not args.no_dp,
        noise_mult=args.noise_mult, max_grad_norm=args.max_grad_norm,
    )
    trainer.train(train_loader=train_loader, val_loader=val_loader,
                  epochs=args.epochs, save_dir=args.save_dir)
    print(f"\n[Phase II] Done. Checkpoint -> {args.save_dir}/model_best.pt")

if __name__ == "__main__":
    main()
