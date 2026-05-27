#!/usr/bin/env python3
"""
Phase I — Threat Modeling: train clean baseline and generate frozen attack corpus.

Usage:
    python scripts/run_phase1.py --dataset cifar10 --epochs 5
    python scripts/run_phase1.py --dataset mnist   --epochs 3 --batch_size 128
"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from models.cnn     import get_model
from utils.data     import get_dataset
from attacks.corpus import generate_corpus


def train_clean(model, loader, device, epochs, lr=0.01):
    model.train()
    opt     = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    print(f"\n[Phase I] Training clean baseline ({epochs} epochs) ...")
    for ep in range(1, epochs + 1):
        total_loss = correct = total = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            opt.zero_grad()
            out = model(imgs); loss = loss_fn(out, lbls)
            loss.backward(); opt.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == lbls).sum().item()
            total      += imgs.size(0)
        sched.step()
        print(f"  Epoch {ep}/{epochs}  loss={total_loss/total:.4f}  CA={correct/total:.3f}")
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    default="cifar10", choices=["cifar10","mnist"])
    p.add_argument("--eps",        type=float, default=8/255)
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--save_dir",   default="./results/corpus")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, _, test_loader = get_dataset(args.dataset, args.data_dir, args.batch_size)
    arch  = "robust" if args.dataset == "cifar10" else "small"
    model = get_model(arch).to(device)
    model = train_clean(model, train_loader, device, args.epochs)
    model.eval()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{args.save_dir}/model_baseline.pt")
    print(f"  Baseline saved -> {args.save_dir}/model_baseline.pt")

    generate_corpus(model=model, dataloader=test_loader,
                    save_dir=args.save_dir, device=device)
    print("\n[Phase I] Complete.")

if __name__ == "__main__":
    main()
