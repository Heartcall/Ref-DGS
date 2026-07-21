import sys
import unittest
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from recover_ball_batches import recover_batch_manifest
from generate_ball_vggt_priors import batches_for_split, image_paths_for_batch, prediction_payloads
from validate_ball_vggt_priors import prepare_payload, tensor_stats, validate_payload


OFFICIAL_BALL = Path(
    "/data1/liuly/reproduction/ref_dgs_main_tables/20260720_010227/"
    "priors/Ref-NeRF/refnerf/ball"
)


class RecoverBatchManifestTest(unittest.TestCase):
    def test_recovers_exact_twenty_view_batches(self):
        train = recover_batch_manifest(OFFICIAL_BALL / "train" / "depth")
        test = recover_batch_manifest(OFFICIAL_BALL / "test" / "depth")

        self.assertEqual(len(train), 5)
        self.assertEqual(len(test), 10)
        for batch in train + test:
            self.assertEqual(len(batch), 20)
            self.assertEqual([item["slot"] for item in batch], list(range(20)))
        self.assertEqual(
            {item["image_name"] for batch in train for item in batch},
            {f"r_{index}" for index in range(100)},
        )
        self.assertEqual(
            {item["image_name"] for batch in test for item in batch},
            {f"r_{index}" for index in range(200)},
        )


class PayloadValidationTest(unittest.TestCase):
    def test_tensor_stats_reports_distribution_and_zero_ratio(self):
        stats = tensor_stats(torch.tensor([0.0, 1.0, 2.0, 3.0]))

        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 3.0)
        self.assertAlmostEqual(stats["mean"], 1.5)
        self.assertEqual(stats["zero_ratio"], 0.25)
        self.assertEqual(stats["quantiles"]["p50"], 1.5)

    def test_prepare_payload_squeezes_only_singleton_channel_and_clones_storage(self):
        backing_depth = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
        backing_conf = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4) + 1

        payload = prepare_payload(backing_depth[1:2], backing_conf[1:2])

        self.assertEqual(payload["depth_map"].shape, (4, 4))
        self.assertEqual(payload["depth_conf"].shape, (4, 4))
        self.assertEqual(payload["depth_map"].untyped_storage().nbytes(), 4 * 4 * 4)
        self.assertEqual(payload["depth_conf"].untyped_storage().nbytes(), 4 * 4 * 4)
        backing_depth.fill_(-1)
        backing_conf.fill_(-1)
        self.assertTrue((payload["depth_map"] >= 0).all())
        self.assertTrue((payload["depth_conf"] >= 1).all())

    def test_validate_payload_rejects_missing_key(self):
        with self.assertRaisesRegex(ValueError, "depth_conf"):
            validate_payload({"depth_map": torch.ones(4, 4)})

    def test_validate_payload_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            validate_payload(
                {"depth_map": torch.ones(4, 4), "depth_conf": torch.ones(3, 4)}
            )

    def test_validate_payload_rejects_nonfinite_depth(self):
        depth = torch.ones(4, 4)
        depth[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_payload({"depth_map": depth, "depth_conf": torch.ones(4, 4)})

    def test_validate_payload_rejects_negative_confidence(self):
        confidence = torch.ones(4, 4)
        confidence[0, 0] = -0.01
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            validate_payload({"depth_map": torch.ones(4, 4), "depth_conf": confidence})


class GeneratorContractTest(unittest.TestCase):
    def test_batches_for_split_reads_recovered_manifest_schema(self):
        manifest = {"train_batches": [[{"image_name": "r_0", "slot": 0}]]}

        self.assertEqual(batches_for_split(manifest, "train"), manifest["train_batches"])

    def test_prediction_payloads_slices_vggt_batch_and_clones_each_map(self):
        depth = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(1, 3, 1, 4, 5)
        confidence = depth + 1

        payloads = prediction_payloads(depth, confidence, expected_views=3)

        self.assertEqual(len(payloads), 3)
        for payload in payloads:
            self.assertEqual(payload["depth_map"].shape, (4, 5))
            self.assertEqual(payload["depth_conf"].shape, (4, 5))
            self.assertEqual(payload["depth_map"].untyped_storage().nbytes(), 4 * 5 * 4)
            self.assertEqual(payload["depth_conf"].untyped_storage().nbytes(), 4 * 5 * 4)
        depth.fill_(-1)
        confidence.fill_(-1)
        self.assertTrue((payloads[0]["depth_map"] >= 0).all())
        self.assertTrue((payloads[0]["depth_conf"] >= 1).all())

    def test_prediction_payloads_accepts_current_vggt_channels_last_layout(self):
        depth = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(1, 3, 4, 5, 1)
        confidence = depth[..., 0] + 1

        payloads = prediction_payloads(depth, confidence, expected_views=3)

        self.assertEqual(len(payloads), 3)
        self.assertEqual(payloads[0]["depth_map"].shape, (4, 5))
        self.assertEqual(payloads[0]["depth_conf"].shape, (4, 5))

    def test_prediction_payloads_rejects_view_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "expected 20 views"):
            prediction_payloads(
                torch.ones(1, 19, 1, 4, 5),
                torch.ones(1, 19, 1, 4, 5),
                expected_views=20,
            )

    def test_image_paths_follow_manifest_slot_order(self):
        scene_root = Path("/dataset/ball")
        batch = [
            {"slot": 0, "image_name": "r_7"},
            {"slot": 1, "image_name": "r_2"},
        ]

        paths = image_paths_for_batch(batch, scene_root, "train", require_exists=False)

        self.assertEqual(
            paths,
            [scene_root / "train" / "r_7.png", scene_root / "train" / "r_2.png"],
        )


if __name__ == "__main__":
    unittest.main()
