import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image


HERE = Path(__file__).resolve().parents[1]
TOOLS = HERE / "tools"
sys.path.insert(0, str(TOOLS))

from build_scene_manifests import apply_recovered_grouping, build_manifest, make_batches
from generate_scene_vggt_priors import prediction_payloads
from validate_scene_vggt_priors import (
    prepare_payload,
    uniform_indices,
    validate_payload,
    validate_payload_quality,
    validate_tree,
)


class ManifestContractTest(unittest.TestCase):
    def _write_transforms(self, root: Path, split: str, names: list[str]) -> None:
        frames = [
            {"file_path": name, "transform_matrix": [[1, 0, 0, 0]] * 4}
            for name in names
        ]
        (root / f"transforms_{split}.json").write_text(
            json.dumps({"frames": frames}), encoding="utf-8"
        )

    def test_make_batches_is_deterministic_and_keeps_partial_tail(self):
        records = [{"image_name": f"r_{index}"} for index in range(43)]

        batches = make_batches(records, batch_size=20)

        self.assertEqual([len(batch) for batch in batches], [20, 20, 3])
        self.assertEqual(
            [[item["image_name"] for item in batch] for batch in batches],
            [
                [f"r_{index}" for index in range(20)],
                [f"r_{index}" for index in range(20, 40)],
                [f"r_{index}" for index in range(40, 43)],
            ],
        )
        self.assertEqual([item["slot"] for item in batches[-1]], [0, 1, 2])

    def test_builds_shiny_manifest_from_transform_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "train").mkdir()
            (root / "test").mkdir()
            self._write_transforms(root, "train", ["train/r_7", "train/r_2"])
            self._write_transforms(root, "test", ["test/r_7"])

            manifest = build_manifest("ShinySynthetic", "ball", root, batch_size=20)

        self.assertEqual(
            [item["image_name"] for item in manifest["train_batches"][0]],
            ["r_7", "r_2"],
        )
        self.assertEqual(manifest["train_batches"][0][0]["source_relpath"], "train/r_7.png")
        self.assertEqual(manifest["test_batches"][0][0]["split"], "test")
        self.assertNotEqual(
            manifest["train_batches"][0][0]["prior_relpath"],
            manifest["test_batches"][0][0]["prior_relpath"],
        )
        self.assertEqual(manifest["unique_view_count"], 3)
        self.assertFalse(manifest["grouping"]["claimed_author_equivalent"])

    def test_builds_glossy_manifest_with_disjoint_train_test_views(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "rgb").mkdir()
            self._write_transforms(root, "train", ["rgb/1", "rgb/2", "rgb/3"])
            self._write_transforms(root, "test", ["rgb/0", "rgb/8"])

            manifest = build_manifest("GlossySynthetic", "angel", root, batch_size=2)

        self.assertEqual([len(batch) for batch in manifest["train_batches"]], [2, 1])
        self.assertEqual([len(batch) for batch in manifest["test_batches"]], [2])
        self.assertEqual(manifest["split_counts"], {"train": 3, "test": 2})
        self.assertEqual(manifest["unique_view_count"], 5)
        self.assertEqual(
            manifest["train_batches"][0][0]["source_relpath"], "rgb/1.png"
        )
        self.assertEqual(
            manifest["train_batches"][0][0]["prior_relpath"],
            "Glossy/GlossySynthetic/angel_blender/depth/1.pth",
        )

    def test_rejects_split_overlap_that_would_overwrite_one_prior(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "rgb").mkdir()
            self._write_transforms(root, "train", ["rgb/1"])
            self._write_transforms(root, "test", ["rgb/1"])

            with self.assertRaisesRegex(ValueError, "overlap"):
                build_manifest("GlossySynthetic", "angel", root, batch_size=20)

    def test_applies_recovered_ball_batch_membership_without_losing_paths(self):
        manifest = {
            "dataset": "ShinySynthetic",
            "scene": "ball",
            "grouping": {},
            "train_batches": [[
                {"image_name": "r_0", "source_relpath": "train/r_0.png"},
                {"image_name": "r_1", "source_relpath": "train/r_1.png"},
            ]],
            "test_batches": [],
        }
        recovered = {
            "protocol": "recovered_from_intact_official_pickle_metadata",
            "train_batches": [[
                {"image_name": "r_1", "slot": 0},
                {"image_name": "r_0", "slot": 1},
            ]],
            "test_batches": [],
        }

        result = apply_recovered_grouping(manifest, recovered)

        self.assertEqual(
            [item["image_name"] for item in result["train_batches"][0]],
            ["r_1", "r_0"],
        )
        self.assertEqual(result["train_batches"][0][0]["source_relpath"], "train/r_1.png")
        self.assertEqual(result["grouping"]["policy"], recovered["protocol"])
        self.assertFalse(result["grouping"]["claimed_author_equivalent"])


class TensorContractTest(unittest.TestCase):
    def test_layout_normalization_preserves_cloned_compact_maps(self):
        depth = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(1, 3, 4, 5, 1)
        confidence = depth[..., 0] + 1

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

    def test_layout_normalization_accepts_channels_first(self):
        depth = torch.ones(1, 2, 1, 4, 5)
        confidence = torch.ones(1, 2, 4, 5)

        payloads = prediction_payloads(depth, confidence, expected_views=2)

        self.assertEqual(payloads[0]["depth_map"].shape, (4, 5))

    def test_validate_payload_requires_schema_shape_finite_and_nonnegative_confidence(self):
        payload = prepare_payload(torch.ones(1, 4, 5), torch.ones(1, 4, 5))
        self.assertIs(validate_payload(payload), payload)
        with self.assertRaisesRegex(ValueError, "depth_conf"):
            validate_payload({"depth_map": torch.ones(4, 5)})
        with self.assertRaisesRegex(ValueError, "same shape"):
            validate_payload(
                {"depth_map": torch.ones(4, 5), "depth_conf": torch.ones(4, 4)}
            )
        bad_depth = torch.ones(4, 5)
        bad_depth[0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_payload({"depth_map": bad_depth, "depth_conf": torch.ones(4, 5)})
        bad_conf = torch.ones(4, 5)
        bad_conf[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            validate_payload({"depth_map": torch.ones(4, 5), "depth_conf": bad_conf})

    def test_quality_validation_rejects_constant_depth_or_confidence(self):
        with self.assertRaisesRegex(ValueError, "constant depth"):
            validate_payload_quality(
                {"depth_map": torch.ones(4, 5), "depth_conf": torch.arange(20).reshape(4, 5)}
            )
        with self.assertRaisesRegex(ValueError, "constant confidence"):
            validate_payload_quality(
                {"depth_map": torch.arange(20).reshape(4, 5), "depth_conf": torch.ones(4, 5)}
            )

    def test_uniform_visualization_indices_cover_endpoints(self):
        self.assertEqual(uniform_indices(100, 5), [0, 25, 50, 74, 99])
        self.assertEqual(uniform_indices(3, 5), [0, 1, 2])


class TreeValidationTest(ManifestContractTest):
    def test_validates_manifest_bijection_hashes_normals_and_five_visualizations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scene_root = root / "scene"
            prior_root = root / "priors"
            official_prior_root = root / "official"
            visual_root = root / "visuals"
            (scene_root / "rgb").mkdir(parents=True)
            self._write_transforms(scene_root, "train", ["rgb/1", "rgb/2", "rgb/3"])
            self._write_transforms(scene_root, "test", ["rgb/0", "rgb/8"])
            for name in ("0", "1", "2", "3", "8"):
                Image.new("RGBA", (8, 8), (50, 100, 150, 255)).save(
                    scene_root / "rgb" / f"{name}.png"
                )
            manifest = build_manifest("GlossySynthetic", "angel", scene_root, batch_size=2)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            for index, record in enumerate(
                [item for split in ("train", "test") for batch in manifest[f"{split}_batches"] for item in batch]
            ):
                prior_path = prior_root / record["prior_relpath"]
                normal_path = official_prior_root / record["normal_relpath"]
                prior_path.parent.mkdir(parents=True, exist_ok=True)
                normal_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    prepare_payload(
                        torch.arange(20).reshape(4, 5) + index,
                        torch.arange(20).reshape(4, 5) + index + 1,
                    ),
                    prior_path,
                )
                Image.new("RGB", (8, 8), (127, 127, 255)).save(normal_path)

            result = validate_tree(
                SimpleNamespace(
                    scene_root=scene_root,
                    manifest=manifest_path,
                    prior_root=prior_root,
                    official_prior_root=official_prior_root,
                    visualization_root=visual_root,
                    expected_height=4,
                    expected_width=5,
                )
            )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["overall"]["file_count"], 5)
        self.assertEqual(result["overall"]["normal_file_count"], 5)
        self.assertEqual(result["overall"]["visualization_count"], 5)
        self.assertEqual(len(result["file_sha256_manifest"]), 5)
        self.assertEqual(result["splits"]["train"]["file_count"], 3)
        self.assertEqual(result["splits"]["test"]["file_count"], 2)


if __name__ == "__main__":
    unittest.main()
