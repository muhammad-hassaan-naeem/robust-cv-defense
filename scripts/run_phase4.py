#!/usr/bin/env python3
"""
Phase IV — Dual-Metric Evaluation.

Runs BOTH metric systems in one pass:
  Classical:    CA, RA (FGSM/PGD-7/PGD-20), CertAcc, ACR
  ZKP-informed: DPA, IPI, SAS, RSB

Usage:
    python scripts/run_phase4.py --checkpoint results/model_best.pt
    python scripts/run_phase4.py --checkpoint results/model_best.pt \
                                  --corpus_dir results/corpus \
                                  --sigma 0.25 --cert_samples 200
"""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
import argparse, torch
from pathlib import Path

from models.cnn                import get_model
from utils.data                import get_dataset
from defense.sanitization      import SanitizationPipeline
from defense.adversarial_train import AdversarialTrainer
from attacks.corpus            import load_corpus
from evaluation.evaluator      import DualEvaluator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--dataset",       default="cifar10", choices=["cifar10","mnist"])
    p.add_argument("--corpus_dir",    default=None)
    p.add_argument("--sigma",         type=float, default=0.25)
    p.add_argument("--eps",           type=float, default=8/255)
    p.add_argument("--cert_samples",  type=int,   default=200)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--noise_mult",    type=float, default=1.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--data_dir",      default="./data")
    p.add_argument("--save_dir",      default="./results")
    p.add_argument("--no_median",    action="store_true")
    p.add_argument("--no_bilateral", action="store_true")
    p.add_argument("--no_squeeze",   action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, _, test_loader = get_dataset(args.dataset, args.data_dir, args.batch_size)

    arch  = "robust" if args.dataset == "cifar10" else "small"
    model = get_model(arch)
    model = AdversarialTrainer.load_model(args.checkpoint, model, device)
    print(f"Checkpoint loaded: {args.checkpoint}")

    sanitizer = SanitizationPipeline(
        use_median    = not args.no_median,
        use_bilateral = not args.no_bilateral,
        use_squeeze   = not args.no_squeeze,
        use_spatial   = False,
    )
    print(f"Sanitizer: {sanitizer}")

    corpus = None
    if args.corpus_dir and Path(args.corpus_dir).exists():
        corpus = load_corpus(args.corpus_dir)
        print(f"Corpus: {len(corpus)} attack variants loaded")

    evaluator = DualEvaluator(
        model              = model,
        device             = device,
        sanitizer          = sanitizer,
        sigma              = args.sigma,
        eps                = args.eps,
        n_smooth_pred      = 100,
        n_smooth_cert      = 300,
        claimed_noise_mult = args.noise_mult,
        claimed_clip_norm  = args.max_grad_norm,
        checkpoint_path    = args.checkpoint,
    )

    report = evaluator.evaluate(
        test_loader  = test_loader,
        corpus       = corpus,
        model_name   = f"RobustCNN-{args.dataset}",
        dataset_name = args.dataset.upper(),
        cert_samples = args.cert_samples,
        save_dir     = args.save_dir,
    )
    print(f"[Phase IV] Complete. Report -> {args.save_dir}/")

if __name__ == "__main__":
    main()
