"""
tests/test_pipeline.py
----------------------
Full test suite — 52 tests covering:
  models, attacks, sanitization, randomized smoothing,
  classical metrics (CA/RA/CertAcc/ACR), ZKP metrics (DPA/IPI/SAS/RSB),
  and DualReport integration.

Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
import torch
import numpy as np

from models.cnn                import SmallCNN, RobustCNN, get_model
from attacks.fgsm              import fgsm_attack
from attacks.pgd               import pgd_attack
from defense.sanitization      import (
    adaptive_median_filter, bit_depth_squeeze,
    spatial_transform, SanitizationPipeline, EXPECTED_PIPELINE_ORDER,
)
from defense.randomized_smooth import RandomizedSmoother, ABSTAIN
from utils.metrics import (
    DualReport, ZKP_SCOPE_NOTE, EXPECTED_TRANSFORM_ORDER,
    clean_accuracy, robust_accuracy,
    certified_accuracy_and_acr,
    dp_adherence, dp_adherence_from_checkpoint,
    image_hash, compute_reference_hashes, input_provenance_integrity,
    sanitization_audit_score,
    robustness_soundness_bound,
)

DEVICE = torch.device("cpu")


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def small_model():
    m = SmallCNN(num_classes=10); m.eval(); return m

@pytest.fixture
def robust_model():
    m = RobustCNN(num_classes=10); m.eval(); return m

@pytest.fixture
def mnist_batch():
    torch.manual_seed(0)
    return torch.rand(4, 1, 28, 28), torch.randint(0, 10, (4,))

@pytest.fixture
def cifar_batch():
    torch.manual_seed(0)
    return torch.rand(4, 3, 32, 32), torch.randint(0, 10, (4,))

@pytest.fixture
def tiny_loader(mnist_batch):
    imgs, lbls = mnist_batch
    ds = torch.utils.data.TensorDataset(imgs, lbls)
    return torch.utils.data.DataLoader(ds, batch_size=4)


# ── Models ────────────────────────────────────────────────────────────────────
class TestModels:
    def test_small_cnn_shape(self, small_model, mnist_batch):
        assert small_model(mnist_batch[0]).shape == (4, 10)

    def test_robust_cnn_shape(self, robust_model, cifar_batch):
        assert robust_model(cifar_batch[0]).shape == (4, 10)

    def test_factory_small(self):
        assert isinstance(get_model("small"), SmallCNN)

    def test_factory_robust(self):
        assert isinstance(get_model("robust"), RobustCNN)

    def test_factory_invalid(self):
        with pytest.raises(ValueError):
            get_model("unknown")


# ── Attacks ───────────────────────────────────────────────────────────────────
class TestAttacks:
    def test_fgsm_shape(self, small_model, mnist_batch):
        imgs, lbls = mnist_batch
        assert fgsm_attack(small_model, imgs, lbls).shape == imgs.shape

    def test_fgsm_linf_bound(self, small_model, mnist_batch):
        imgs, lbls = mnist_batch; eps = 8/255
        adv = fgsm_attack(small_model, imgs, lbls, eps=eps)
        assert (adv - imgs).abs().max().item() <= eps + 1e-6

    def test_fgsm_clipped(self, small_model, mnist_batch):
        imgs, lbls = mnist_batch
        adv = fgsm_attack(small_model, imgs, lbls)
        assert adv.min() >= -1e-6 and adv.max() <= 1.0 + 1e-6

    def test_pgd_linf_shape(self, small_model, mnist_batch):
        imgs, lbls = mnist_batch
        adv = pgd_attack(small_model, imgs, lbls, steps=5)
        assert adv.shape == imgs.shape

    def test_pgd_linf_bound(self, small_model, mnist_batch):
        imgs, lbls = mnist_batch; eps = 8/255
        adv = pgd_attack(small_model, imgs, lbls, eps=eps, steps=5)
        assert (adv - imgs).abs().max().item() <= eps + 1e-5

    def test_pgd_l2_bound(self, small_model, mnist_batch):
        imgs, lbls = mnist_batch; eps = 0.5
        adv = pgd_attack(small_model, imgs, lbls, eps=eps, steps=5, norm="L2")
        l2  = (adv - imgs).view(imgs.size(0), -1).norm(p=2, dim=1).max().item()
        assert l2 <= eps + 1e-4

    def test_pgd_clipped(self, small_model, mnist_batch):
        imgs, lbls = mnist_batch
        adv = pgd_attack(small_model, imgs, lbls, steps=5)
        assert adv.min() >= -1e-6 and adv.max() <= 1.0 + 1e-6


# ── Sanitization ──────────────────────────────────────────────────────────────
class TestSanitization:
    def test_median_shape(self, cifar_batch):
        imgs, _ = cifar_batch
        assert adaptive_median_filter(imgs).shape == imgs.shape

    def test_median_range(self, cifar_batch):
        imgs, _ = cifar_batch
        out = adaptive_median_filter(imgs)
        assert out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6

    def test_squeeze_shape(self, cifar_batch):
        imgs, _ = cifar_batch
        assert bit_depth_squeeze(imgs, bits=5).shape == imgs.shape

    def test_squeeze_quantized(self, cifar_batch):
        imgs, _ = cifar_batch
        out  = bit_depth_squeeze(imgs, bits=5)
        vals = (out * 31).round() / 31
        assert (out - vals).abs().max().item() < 1e-5

    def test_spatial_shape(self, cifar_batch):
        imgs, _ = cifar_batch
        assert spatial_transform(imgs, seed=42).shape == imgs.shape

    def test_spatial_range(self, cifar_batch):
        imgs, _ = cifar_batch
        out = spatial_transform(imgs, seed=42)
        assert out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6

    def test_pipeline_shape(self, cifar_batch):
        imgs, _ = cifar_batch
        p   = SanitizationPipeline(use_median=True, use_bilateral=True,
                                    use_squeeze=True, use_spatial=False)
        out = p(imgs)
        assert out.shape == imgs.shape and out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6

    def test_pipeline_reduces_unique_values(self, small_model, mnist_batch):
        imgs, lbls = mnist_batch
        adv = fgsm_attack(small_model, imgs, lbls, eps=32/255)
        p   = SanitizationPipeline(use_median=True, use_bilateral=False,
                                    use_squeeze=True, use_spatial=False)
        san = p(adv)
        assert san.shape == adv.shape
        assert san.view(-1).unique().numel() <= adv.view(-1).unique().numel()

    def test_expected_pipeline_order(self):
        assert "adaptive_median_filter" in EXPECTED_PIPELINE_ORDER
        assert "bit_depth_squeeze" in EXPECTED_PIPELINE_ORDER


# ── Randomized Smoothing ──────────────────────────────────────────────────────
class TestRandomizedSmoothing:
    def test_predict_shape(self, small_model, mnist_batch):
        imgs, _ = mnist_batch
        s = RandomizedSmoother(small_model, sigma=0.1, device=DEVICE, n_samples=10)
        assert s.predict(imgs, n_samples=10).shape == (4,)

    def test_predict_valid_classes(self, small_model, mnist_batch):
        imgs, _ = mnist_batch
        s = RandomizedSmoother(small_model, sigma=0.1, device=DEVICE, n_samples=10)
        preds = s.predict(imgs, n_samples=10)
        assert (preds >= 0).all() and (preds < 10).all()

    def test_certify_returns_tuple(self, small_model, mnist_batch):
        imgs, _ = mnist_batch
        s = RandomizedSmoother(small_model, sigma=0.1, device=DEVICE,
                               n_samples=20, n_samples_cert=50)
        pred, radius = s.certify(imgs[0], n_samples=50)
        assert isinstance(pred, int) and isinstance(radius, float) and radius >= 0.0

    def test_certify_radius_nonneg(self, small_model, mnist_batch):
        imgs, _ = mnist_batch
        s = RandomizedSmoother(small_model, sigma=0.25, device=DEVICE, n_samples_cert=100)
        for i in range(imgs.size(0)):
            pred, r = s.certify(imgs[i], n_samples=100)
            assert r >= 0.0 if pred != ABSTAIN else r == 0.0

    def test_abstain_is_minus_one(self):
        assert ABSTAIN == -1

    def test_higher_sigma_larger_radius(self, small_model, mnist_batch):
        imgs, _ = mnist_batch
        radii = {0.12: [], 0.5: []}
        for sigma in radii:
            s = RandomizedSmoother(small_model, sigma=sigma, device=DEVICE, n_samples_cert=100)
            for i in range(imgs.size(0)):
                _, r = s.certify(imgs[i], n_samples=100)
                if r > 0: radii[sigma].append(r)
        if radii[0.12] and radii[0.5]:
            assert np.mean(radii[0.5]) >= np.mean(radii[0.12]) - 0.1


# ── Classical Metric: CA ──────────────────────────────────────────────────────
class TestClassicalCA:
    def test_ca_range(self, small_model, tiny_loader):
        ca = clean_accuracy(small_model, tiny_loader, DEVICE)
        assert 0.0 <= ca <= 1.0

    def test_ca_with_sanitizer(self, small_model, tiny_loader):
        san = SanitizationPipeline(use_median=True, use_bilateral=False,
                                    use_squeeze=True, use_spatial=False)
        ca  = clean_accuracy(small_model, tiny_loader, DEVICE, sanitizer=san)
        assert 0.0 <= ca <= 1.0

    def test_ca_deterministic(self, small_model, tiny_loader):
        ca1 = clean_accuracy(small_model, tiny_loader, DEVICE)
        ca2 = clean_accuracy(small_model, tiny_loader, DEVICE)
        assert ca1 == ca2


# ── Classical Metric: RA ──────────────────────────────────────────────────────
class TestClassicalRA:
    def test_ra_range(self, small_model, tiny_loader):
        atk = lambda m, imgs, lbls: fgsm_attack(m, imgs, lbls, eps=8/255)
        ra  = robust_accuracy(small_model, tiny_loader, atk, DEVICE)
        assert 0.0 <= ra <= 1.0

    def test_ra_leq_ca(self, small_model, tiny_loader):
        ca  = clean_accuracy(small_model, tiny_loader, DEVICE)
        atk = lambda m, imgs, lbls: pgd_attack(m, imgs, lbls, eps=8/255, steps=7)
        ra  = robust_accuracy(small_model, tiny_loader, atk, DEVICE)
        assert ra <= ca + 0.05  # PGD-RA should not exceed CA


# ── Classical Metrics: CertAcc + ACR ─────────────────────────────────────────
class TestCertAccACR:
    def _make_cert_results(self, n_correct, n_abstain, n_wrong, radius=0.3):
        from defense.randomized_smooth import ABSTAIN
        r = []
        for _ in range(n_correct): r.append({"true": 1, "pred": 1, "radius": radius})
        for _ in range(n_abstain): r.append({"true": 1, "pred": ABSTAIN, "radius": 0.0})
        for _ in range(n_wrong):   r.append({"true": 1, "pred": 2, "radius": radius})
        return r

    def test_perfect(self):
        results = [{"true": i, "pred": i, "radius": 0.3} for i in range(10)]
        ca, acr, ab = certified_accuracy_and_acr(results)
        assert ca == 1.0 and acr > 0 and ab == 0.0

    def test_empty(self):
        ca, acr, ab = certified_accuracy_and_acr([])
        assert ca == 0.0 and acr == 0.0 and ab == 0.0

    def test_range(self):
        results = self._make_cert_results(3, 1, 1)
        ca, acr, ab = certified_accuracy_and_acr(results)
        assert 0.0 <= ca <= 1.0 and acr >= 0.0 and 0.0 <= ab <= 1.0

    def test_wrong_excluded(self):
        results = self._make_cert_results(n_correct=3, n_abstain=0, n_wrong=7)
        ca, _, _ = certified_accuracy_and_acr(results)
        assert ca == 0.3

    def test_acr_mean_correct(self):
        results = [{"true": 0, "pred": 0, "radius": 0.4}] * 5
        _, acr, _ = certified_accuracy_and_acr(results)
        assert abs(acr - 0.4) < 1e-6


# ── ZKP Metric 1: DPA ─────────────────────────────────────────────────────────
class TestDPA:
    def test_exact_match(self):
        assert dp_adherence(1.1, 1.0, 1.1, 1.0) == 1.0

    def test_noise_mismatch(self):
        assert dp_adherence(1.1, 1.0, 0.5, 1.0) == 0.0

    def test_clip_mismatch(self):
        assert dp_adherence(1.1, 1.0, 1.1, 5.0) == 0.0

    def test_within_tolerance(self):
        assert dp_adherence(1.1, 1.0, 1.1005, 1.0005, tolerance=1e-3) == 1.0

    def test_output_binary(self):
        for nm in [0.5, 1.1, 2.0]:
            assert dp_adherence(1.1, 1.0, nm, 1.0) in (0.0, 1.0)

    def test_zkp_scope_note_present(self):
        assert "out-of-scope" in ZKP_SCOPE_NOTE.lower()
        assert len(ZKP_SCOPE_NOTE) > 50


# ── ZKP Metric 2: IPI ─────────────────────────────────────────────────────────
class TestIPI:
    def test_hash_deterministic(self, cifar_batch):
        imgs, _ = cifar_batch
        assert image_hash(imgs[0]) == image_hash(imgs[0])

    def test_hash_different(self, cifar_batch):
        imgs, _ = cifar_batch
        assert image_hash(imgs[0]) != image_hash(imgs[1])

    def test_hash_is_sha256(self, cifar_batch):
        imgs, _ = cifar_batch
        assert len(image_hash(imgs[0])) == 64

    def test_ipi_perfect(self, cifar_batch):
        imgs, _ = cifar_batch
        refs = compute_reference_hashes(imgs)
        assert input_provenance_integrity(imgs, refs) == 1.0

    def test_ipi_tampered(self, cifar_batch):
        imgs, _ = cifar_batch
        refs     = compute_reference_hashes(imgs)
        tampered = imgs.clone(); tampered[0] += 0.01
        assert input_provenance_integrity(tampered, refs) < 1.0

    def test_ipi_all_tampered(self, cifar_batch):
        imgs, _ = cifar_batch
        refs = compute_reference_hashes(imgs)
        assert input_provenance_integrity(imgs + 0.01, refs) == 0.0

    def test_ref_hashes_length(self, cifar_batch):
        imgs, _ = cifar_batch
        assert len(compute_reference_hashes(imgs)) == imgs.size(0)


# ── ZKP Metric 3: SAS ─────────────────────────────────────────────────────────
class TestSAS:
    FULL = ["adaptive_median_filter", "bilateral_denoise", "bit_depth_squeeze"]
    PART = ["adaptive_median_filter", "bit_depth_squeeze"]

    def test_perfect(self):
        assert sanitization_audit_score([self.FULL] * 5) == 1.0

    def test_missing_transform(self):
        assert sanitization_audit_score([self.PART] * 4) == 0.0

    def test_empty_log(self):
        assert sanitization_audit_score([]) == 0.0

    def test_partial(self):
        log   = [self.FULL, self.FULL, self.PART, self.PART]
        score = sanitization_audit_score(log)
        assert score == 0.5

    def test_range(self):
        s = sanitization_audit_score([self.FULL])
        assert 0.0 <= s <= 1.0

    def test_expected_order_constant(self):
        assert isinstance(EXPECTED_TRANSFORM_ORDER, list) and len(EXPECTED_TRANSFORM_ORDER) >= 2


# ── ZKP Metric 4: RSB ─────────────────────────────────────────────────────────
class TestRSB:
    def _results(self, n_correct, n_abstain, n_wrong, radius=0.3):
        r = []
        for _ in range(n_correct): r.append({"true": 1, "pred": 1, "radius": radius})
        for _ in range(n_abstain): r.append({"true": 1, "pred": ABSTAIN, "radius": 0.0})
        for _ in range(n_wrong):   r.append({"true": 1, "pred": 2, "radius": radius})
        return r

    def test_perfect(self):
        results = [{"true": i, "pred": i, "radius": 0.3} for i in range(10)]
        rsb, _, _ = robustness_soundness_bound(results)
        assert rsb == 1.0

    def test_empty(self):
        rsb, mr, ab = robustness_soundness_bound([])
        assert rsb == 0.0 and mr == 0.0 and ab == 0.0

    def test_all_abstain(self):
        results = [{"true": 1, "pred": ABSTAIN, "radius": 0.0}] * 5
        rsb, _, ab = robustness_soundness_bound(results)
        assert rsb == 0.0 and ab == 1.0

    def test_range(self):
        r = self._results(3, 1, 1)
        rsb, mr, ab = robustness_soundness_bound(r)
        assert 0.0 <= rsb <= 1.0 and mr >= 0.0 and 0.0 <= ab <= 1.0

    def test_wrong_excluded(self):
        r   = self._results(3, 0, 7)
        rsb, _, _ = robustness_soundness_bound(r)
        assert rsb == 0.3

    def test_mean_radius_correct(self):
        results = [{"true": 0, "pred": 0, "radius": 0.4}] * 4 + \
                  [{"true": 0, "pred": ABSTAIN, "radius": 0.0}]
        _, mr, _ = robustness_soundness_bound(results)
        assert abs(mr - 0.4) < 1e-6

    def test_zero_radius_excluded(self):
        results = [
            {"true": 1, "pred": 1, "radius": 0.0},
            {"true": 1, "pred": 1, "radius": 0.3},
        ]
        rsb, _, _ = robustness_soundness_bound(results)
        assert rsb == 0.5


# ── DualReport integration ────────────────────────────────────────────────────
class TestDualReport:
    def _make_report(self):
        return DualReport(
            model_name="TestCNN", dataset="MNIST", eps=8/255, sigma=0.25,
            ca=0.9, ra={"FGSM": 0.7, "PGD-7": 0.6}, cert_acc=0.5, acr=0.35,
            classical_abstain_rate=0.0,
            dpa=1.0, ipi=1.0, sas=1.0, rsb=0.5,
            rsb_mean_radius=0.35, zkp_abstain_rate=0.0,
        )

    def test_instantiation(self):
        r = self._make_report()
        assert r.ca == 0.9 and r.dpa == 1.0

    def test_summary_dict_keys(self):
        r = self._make_report()
        sd = r.summary_dict()
        assert "classical/ca" in sd
        assert "zkp/dpa"      in sd
        assert "zkp/rsb"      in sd
        assert "classical/ra_fgsm" in sd

    def test_cert_acc_equals_rsb(self):
        # CertAcc and RSB should be numerically equal — they both measure
        # fraction correctly certified. Difference is framing only.
        r = self._make_report()
        assert r.cert_acc == r.rsb


# ── AutoAttack Tests ──────────────────────────────────────────────────────────
class TestAutoAttack:
    """Tests for the AutoAttack ensemble and its components."""

    def test_apgd_ce_shape(self, small_model, mnist_batch):
        from attacks.autoattack import APGD
        imgs, lbls = mnist_batch
        apgd = APGD(small_model, eps=8/255, steps=5, loss="ce", device=DEVICE)
        adv  = apgd.perturb(imgs, lbls)
        assert adv.shape == imgs.shape

    def test_apgd_dlr_shape(self, small_model, mnist_batch):
        from attacks.autoattack import APGD
        imgs, lbls = mnist_batch
        apgd = APGD(small_model, eps=8/255, steps=5, loss="dlr", device=DEVICE)
        adv  = apgd.perturb(imgs, lbls)
        assert adv.shape == imgs.shape

    def test_apgd_linf_bound(self, small_model, mnist_batch):
        from attacks.autoattack import APGD
        imgs, lbls = mnist_batch; eps = 8/255
        apgd = APGD(small_model, eps=eps, steps=5, loss="ce", device=DEVICE)
        adv  = apgd.perturb(imgs, lbls)
        assert (adv - imgs).abs().max().item() <= eps + 1e-5

    def test_apgd_clipped(self, small_model, mnist_batch):
        from attacks.autoattack import APGD
        imgs, lbls = mnist_batch
        apgd = APGD(small_model, eps=8/255, steps=5, device=DEVICE)
        adv  = apgd.perturb(imgs, lbls)
        assert adv.min() >= -1e-5 and adv.max() <= 1.0 + 1e-5

    def test_square_attack_shape(self, small_model, mnist_batch):
        from attacks.autoattack import SquareAttack
        imgs, lbls = mnist_batch
        sq  = SquareAttack(small_model, eps=8/255, n_queries=10, device=DEVICE)
        adv = sq.perturb(imgs, lbls)
        assert adv.shape == imgs.shape

    def test_square_attack_linf_bound(self, small_model, mnist_batch):
        from attacks.autoattack import SquareAttack
        imgs, lbls = mnist_batch; eps = 8/255
        sq  = SquareAttack(small_model, eps=eps, n_queries=10, device=DEVICE)
        adv = sq.perturb(imgs, lbls)
        assert (adv - imgs).abs().max().item() <= eps + 1e-5

    def test_square_attack_clipped(self, small_model, mnist_batch):
        from attacks.autoattack import SquareAttack
        imgs, lbls = mnist_batch
        sq  = SquareAttack(small_model, eps=8/255, n_queries=10, device=DEVICE)
        adv = sq.perturb(imgs, lbls)
        assert adv.min() >= -1e-5 and adv.max() <= 1.0 + 1e-5

    def test_ensemble_shape(self, small_model, mnist_batch):
        from attacks.autoattack import AutoAttackEnsemble
        imgs, lbls = mnist_batch
        aa  = AutoAttackEnsemble(small_model, eps=8/255, steps=3,
                                  n_queries=10, device=DEVICE, verbose=False)
        adv = aa.perturb(imgs, lbls)
        assert adv.shape == imgs.shape

    def test_ensemble_linf_bound(self, small_model, mnist_batch):
        from attacks.autoattack import AutoAttackEnsemble
        imgs, lbls = mnist_batch; eps = 8/255
        aa  = AutoAttackEnsemble(small_model, eps=eps, steps=3,
                                  n_queries=10, device=DEVICE, verbose=False)
        adv = aa.perturb(imgs, lbls)
        assert (adv - imgs).abs().max().item() <= eps + 1e-5

    def test_ensemble_clipped(self, small_model, mnist_batch):
        from attacks.autoattack import AutoAttackEnsemble
        imgs, lbls = mnist_batch
        aa  = AutoAttackEnsemble(small_model, eps=8/255, steps=3,
                                  n_queries=10, device=DEVICE, verbose=False)
        adv = aa.perturb(imgs, lbls)
        assert adv.min() >= -1e-5 and adv.max() <= 1.0 + 1e-5

    def test_autoattack_ra_range(self, small_model, tiny_loader):
        from attacks.autoattack import autoattack_accuracy
        ra = autoattack_accuracy(small_model, tiny_loader, eps=8/255,
                                  steps=3, n_queries=10, device=DEVICE, verbose=False)
        assert 0.0 <= ra <= 1.0

    def test_dlr_loss_shape(self, small_model, mnist_batch):
        from attacks.autoattack import dlr_loss
        imgs, lbls = mnist_batch
        with torch.no_grad():
            logits = small_model(imgs)
        loss = dlr_loss(logits, lbls)
        assert loss.shape == ()   # scalar

    def test_autoattack_leq_pgd(self, small_model, tiny_loader):
        """AA-RA should be ≤ PGD-7-RA (AA is strictly stronger)."""
        from attacks.autoattack import autoattack_accuracy
        ra_aa  = autoattack_accuracy(small_model, tiny_loader, eps=8/255,
                                      steps=3, n_queries=10, device=DEVICE, verbose=False)
        ra_pgd = robust_accuracy(
            small_model, tiny_loader,
            lambda m, imgs, lbls: pgd_attack(m, imgs, lbls, eps=8/255, steps=7),
            DEVICE
        )
        # AA ≤ PGD + small tolerance (stochastic)
        assert ra_aa <= ra_pgd + 0.15


# ── Trade-off Curve Tests ─────────────────────────────────────────────────────
class TestTradeoffCurves:
    """Tests for the accuracy–robustness trade-off plotting module."""

    def _make_points(self):
        from analysis.tradeoff_curves import SigmaPoint
        return [
            SigmaPoint(sigma=0.12, ca=0.85, ra_fgsm=0.70, ra_pgd7=0.60,
                       cert_acc=0.40, acr=0.15, radii=[0.1, 0.2, 0.0, 0.15]),
            SigmaPoint(sigma=0.25, ca=0.80, ra_fgsm=0.65, ra_pgd7=0.55,
                       cert_acc=0.50, acr=0.30, radii=[0.2, 0.3, 0.25, 0.0]),
            SigmaPoint(sigma=0.50, ca=0.70, ra_fgsm=0.55, ra_pgd7=0.45,
                       cert_acc=0.55, acr=0.55, radii=[0.4, 0.5, 0.6, 0.0]),
        ]

    def test_sigma_point_creation(self):
        from analysis.tradeoff_curves import SigmaPoint
        sp = SigmaPoint(sigma=0.25, ca=0.8, cert_acc=0.5, acr=0.3)
        assert sp.sigma == 0.25 and sp.ca == 0.8

    def test_plot_tradeoff_runs(self, tmp_path):
        from analysis.tradeoff_curves import plot_tradeoff_curve
        pts = self._make_points()
        plot_tradeoff_curve(pts, tmp_path, title="Test", show_aa=False)
        assert (tmp_path / "fig1_tradeoff_curve.png").exists()

    def test_plot_radius_dist_runs(self, tmp_path):
        from analysis.tradeoff_curves import plot_radius_distribution
        pts = self._make_points()
        plot_radius_distribution(pts, tmp_path, title="Test")
        assert (tmp_path / "fig2_radius_distribution.png").exists()

    def test_plot_ablation_runs(self, tmp_path):
        from analysis.tradeoff_curves import plot_ablation, AblationPoint
        pts = [
            AblationPoint("No Defense",  0.85, 0.35, 0.0,  0.0),
            AblationPoint("L1 only",     0.84, 0.40, 0.0,  0.0),
            AblationPoint("Full",        0.80, 0.58, 0.50, 0.30),
        ]
        plot_ablation(pts, tmp_path, title="Test Ablation")
        assert (tmp_path / "fig3_ablation.png").exists()

    def test_save_load_sigma_sweep(self, tmp_path):
        from analysis.tradeoff_curves import save_sigma_sweep, load_sigma_sweep
        pts  = self._make_points()
        path = tmp_path / "sweep.json"
        save_sigma_sweep(pts, path)
        loaded = load_sigma_sweep(path)
        assert len(loaded) == len(pts)
        assert loaded[0].sigma == pts[0].sigma
        assert loaded[1].ca    == pts[1].ca

    def test_sigma_point_radii_stored(self):
        from analysis.tradeoff_curves import SigmaPoint
        sp = SigmaPoint(sigma=0.25, ca=0.8, cert_acc=0.5, acr=0.3,
                        radii=[0.1, 0.2, 0.3])
        assert len(sp.radii) == 3 and sp.radii[1] == 0.2


# ── Ablation Tests ────────────────────────────────────────────────────────────
class TestAblation:
    """Tests for the ablation study module."""

    def test_ablation_configs_complete(self):
        from analysis.ablation import ABLATION_CONFIGS
        names = [c["name"] for c in ABLATION_CONFIGS]
        assert "No Defense"      in names
        assert "Full (L1+L2+L3)" in names
        assert len(ABLATION_CONFIGS) >= 6

    def test_ablation_point_creation(self):
        from analysis.tradeoff_curves import AblationPoint
        pt = AblationPoint(label="Test", ca=0.85, ra_pgd7=0.6,
                            cert_acc=0.5, acr=0.35)
        assert pt.label == "Test" and pt.ca == 0.85

    def test_ablation_colors_defined(self):
        from analysis.ablation import ABLATION_CONFIGS
        for cfg in ABLATION_CONFIGS:
            assert "color" in cfg
            assert cfg["color"].startswith("#")

    def test_load_ablation_results(self, tmp_path):
        import json
        from analysis.ablation import load_ablation_results
        data = [{"label": "No Defense", "ca": 0.85, "ra_pgd7": 0.35,
                  "cert_acc": 0.0, "acr": 0.0, "color": "#475569"},
                {"label": "Full",       "ca": 0.80, "ra_pgd7": 0.58,
                  "cert_acc": 0.50, "acr": 0.30, "color": "#34d399"}]
        path = tmp_path / "ablation.json"
        with open(path, "w") as f: json.dump(data, f)
        pts = load_ablation_results(path)
        assert len(pts) == 2
        assert pts[0].label == "No Defense"
        assert pts[1].cert_acc == 0.50
