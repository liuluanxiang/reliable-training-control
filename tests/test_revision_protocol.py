import unittest
import json
import tempfile
from pathlib import Path

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from run_experiment import planned_runs, scientific_run_hash, train_one_epoch
from src.controller import ControllerConfig, ReliabilityController
from src.data import _split_indices, build_loaders, image_transform
from src.models import build_model
from src.signals import ReliabilitySignals, SignalConfig
from revision_experiments.scripts.analyze_revision import paired_statistics, shared_analysis_rows


ROOT = Path(__file__).resolve().parents[1]
DEIT_CHECKPOINT = ROOT / "data" / "pretrained" / "deit_tiny_patch16_224-a1311bcf.pth"
TINY_ROOT = ROOT / "data" / "tiny-imagenet-200"


class RevisionControllerTests(unittest.TestCase):
    def test_lr_counter_does_not_prevent_reliability_stopping(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        controller = ReliabilityController(
            ControllerConfig(gamma=0.5, min_lr=1e-5, p_lr=2, p_stop=5, epsilon=1e-4)
        )

        controller.step(optimizer, rbar=1.0, val_loss=1.0, epoch=1)
        action = None
        for epoch in range(2, 7):
            action = controller.step(
                optimizer, rbar=1.0, val_loss=1.0 - 0.01 * epoch, epoch=epoch
            )

        self.assertTrue(action["early_stop"])
        self.assertEqual(action["stop_reason"], "reliability_stagnation")
        self.assertEqual(action["bad_rel_stop"], 5)
        self.assertEqual(action["bad_rel"], action["bad_rel_stop"])
        self.assertEqual(action["lr_reductions"], 2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.025)

    def test_minimum_lr_is_not_counted_as_a_reduction(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1e-5)
        controller = ReliabilityController(
            ControllerConfig(gamma=0.5, min_lr=1e-5, p_lr=1, p_stop=3, epsilon=1e-4)
        )
        controller.step(optimizer, rbar=1.0, val_loss=1.0, epoch=1)
        action = controller.step(optimizer, rbar=1.0, val_loss=0.9, epoch=2)
        self.assertFalse(action["lr_reduced"])
        self.assertEqual(action["lr_reductions"], 0)


class RevisionTrainingLoopTests(unittest.TestCase):
    def test_gradient_norm_is_computed_once_on_final_minibatch(self):
        class CountingSignals:
            def __init__(self):
                self.gradient_calls = 0

            def begin_epoch(self, model):
                return None

            def gradient_norm(self, model):
                self.gradient_calls += 1
                return 1.0

            def update_norm(self, model):
                return 2.0

        model = torch.nn.Linear(4, 2)
        loader = DataLoader(
            TensorDataset(torch.randn(6, 4), torch.tensor([0, 1, 0, 1, 0, 1])), batch_size=2
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        signals = CountingSignals()
        result = train_one_epoch(
            model, loader, optimizer, torch.nn.CrossEntropyLoss(), torch.device("cpu"), signals
        )
        self.assertEqual(signals.gradient_calls, 1)
        self.assertEqual(result["grad_norm"], 1.0)
        self.assertEqual(result["update_norm"], 2.0)

    def test_existing_result_reuse_requires_scientific_hash_match(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {
                "experiment_name": "hash_test", "experiment_type": "test", "dataset": "cifar10",
                "model": "resnet18", "methods": ["step"], "seeds": [1],
                "results_dir": directory, "lr": 0.1,
            }
            output = Path(directory) / "hash_test" / "step" / "seed_1"
            output.mkdir(parents=True)
            for artifact in [
                "best_checkpoint.pt", "config_used.json", "run_metadata.json", "history.csv",
                "test_logits.pt", "test_targets.pt", "calibration_bins.csv",
                "risk_coverage_curve.csv",
            ]:
                (output / artifact).touch()
            summary = {
                "method": "step", "seed": 1,
                "scientific_config_hash": scientific_run_hash({**cfg, "lr": 0.2}, "step", 1),
            }
            (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(planned_runs(cfg, "rerun-partial")[0]["status"], "archive_mismatch_then_run")
            self.assertEqual(planned_runs(cfg, "skip")[0]["status"], "error_config_mismatch")

            summary["scientific_config_hash"] = scientific_run_hash(cfg, "step", 1)
            (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(planned_runs(cfg, "rerun-partial")[0]["status"], "skip_completed")

    def test_summary_without_full_artifact_set_is_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {
                "experiment_name": "partial_test", "experiment_type": "test",
                "dataset": "cifar10", "model": "resnet18", "methods": ["step"],
                "seeds": [1], "results_dir": directory, "lr": 0.1,
            }
            output = Path(directory) / "partial_test" / "step" / "seed_1"
            output.mkdir(parents=True)
            summary = {
                "method": "step", "seed": 1,
                "scientific_config_hash": scientific_run_hash(cfg, "step", 1),
            }
            (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(planned_runs(cfg, "skip")[0]["status"], "skip_partial")
            self.assertEqual(
                planned_runs(cfg, "rerun-partial")[0]["status"], "archive_partial_then_run"
            )

    def test_signal_norms_match_direct_parameter_calculation(self):
        model = torch.nn.Linear(2, 1, bias=False)
        signals = ReliabilitySignals(model, n_train=10, cfg=SignalConfig(prior_sigma=0.5))
        signals.begin_epoch(model)
        model.weight.grad = torch.tensor([[3.0, 4.0]])
        self.assertAlmostEqual(signals.gradient_norm(model), 5.0, places=6)
        with torch.no_grad():
            model.weight.add_(torch.tensor([[0.3, 0.4]]))
        self.assertAlmostEqual(signals.update_norm(model), 0.5, places=6)
        self.assertAlmostEqual(signals.complexity_ct(model), 0.5, places=6)


class RevisionDataTests(unittest.TestCase):
    def test_split_is_reproducible_and_disjoint(self):
        train_a, val_a = _split_indices(100, 0.1, 7)
        train_b, val_b = _split_indices(100, 0.1, 7)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertFalse(set(train_a) & set(val_a))
        self.assertEqual(len(train_a), 90)
        self.assertEqual(len(val_a), 10)

    def test_cifar_validation_transform_is_deterministic(self):
        transform = image_transform("cifar100", train=False)
        names = [type(item).__name__ for item in transform.transforms]
        self.assertNotIn("RandomCrop", names)
        self.assertNotIn("RandomHorizontalFlip", names)
        image = Image.new("RGB", (32, 32), color=(37, 89, 151))
        self.assertTrue(torch.equal(transform(image), transform(image)))

    @unittest.skipUnless(TINY_ROOT.exists(), "Tiny ImageNet has not been extracted")
    def test_tiny_imagenet_partition_counts(self):
        train, validation, test, classes = build_loaders(
            "tiny_imagenet", str(ROOT / "data"), 32, 0, 0.1, 1,
            image_size=64, normalization="imagenet",
        )
        self.assertEqual(classes, 200)
        self.assertEqual(len(train.dataset), 90000)
        self.assertEqual(len(validation.dataset), 10000)
        self.assertEqual(len(test.dataset), 10000)
        self.assertFalse(set(train.dataset.indices) & set(validation.dataset.indices))


class RevisionModelTests(unittest.TestCase):
    @unittest.skipUnless(DEIT_CHECKPOINT.exists(), "Official DeiT checkpoint has not been downloaded")
    def test_official_deit_checkpoint_loads_with_new_head(self):
        torch.manual_seed(1)
        model = build_model(
            "deit_tiny_patch16_224", 100, pretrained_checkpoint=str(DEIT_CHECKPOINT)
        )
        self.assertEqual(model.head.in_features, 192)
        self.assertEqual(model.head.out_features, 100)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(parameter_count, 5_000_000)
        self.assertLess(parameter_count, 6_000_000)


class RevisionAnalysisTests(unittest.TestCase):
    def test_five_seed_statistics_include_all_four_baselines(self):
        rows = []
        methods = ["step", "cosine", "plateau", "reliability_val_loss_only", "reliability"]
        for method_index, method in enumerate(methods):
            for seed in range(1, 6):
                rows.append({
                    "experiment": "revision_main_cifar10_resnet18", "experiment_type": "revision_main",
                    "dataset": "cifar10", "model": "resnet18", "method": method, "seed": seed,
                    "test_acc": 0.7 + 0.01 * method_index + 0.001 * seed,
                })
        frame = pd.DataFrame(rows)
        stats = paired_statistics(frame)
        self.assertEqual(set(stats["baseline"]), {
            "step", "cosine", "plateau", "reliability_val_loss_only"
        })
        self.assertTrue((stats["n_pairs"] == 5).all())
        self.assertTrue((stats["analysis_scope"] == "headline_5_seed").all())
        self.assertTrue(stats["paired_t_p_holm"].notna().all())

    def test_main_rows_are_shared_without_duplicate_training(self):
        rows = []
        for method in ["step", "cosine", "plateau", "reliability_val_loss_only", "reliability"]:
            for seed in range(1, 6):
                rows.append({
                    "experiment": "revision_main_cifar10_resnet18", "experiment_type": "revision_main",
                    "dataset": "cifar10", "model": "resnet18", "method": method, "seed": seed,
                    "budget_mode": "adaptive", "prior_sigma": 0.05,
                })
        shared = shared_analysis_rows(pd.DataFrame(rows))
        self.assertEqual(len(shared), 35)
        self.assertTrue((shared["analysis_source"] == "shared_corrected_main_run").all())


if __name__ == "__main__":
    unittest.main()
