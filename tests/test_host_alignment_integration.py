"""Registered node output seam; SAM3 unpack is the only external boundary double."""
from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

import torch

from test_scail2_colored_mask_node import fake_sam3_unpack_masks, import_root_package


class HostAlignmentIntegrationTests(unittest.TestCase):
    def test_sam3_tracks_flow_through_registered_condition_and_native_adapter_nodes(self) -> None:
        package = import_root_package()
        self.assertEqual(13, len(package.NODE_CLASS_MAPPINGS))
        colored = package.NODE_CLASS_MAPPINGS["SCAILPose2ColoredMask"]()
        condition_node = package.NODE_CLASS_MAPPINGS["SCAILPose2SCAIL2Condition"]()
        adapter = package.NODE_CLASS_MAPPINGS["SCAILPose2WanVideoSCAIL2Adapter"]()
        unpacked = torch.zeros((5, 1, 2, 4), dtype=torch.bool)
        unpacked[..., :2] = True
        native = torch.zeros((5, 1, 16, 32), dtype=torch.bool)
        native[..., :16] = True
        tracks = {
            "packed": {"packed_masks": torch.ones((5, 1, 2, 1), dtype=torch.uint8)},
            "native": {"masks": native},
            "empty": {"packed_masks": None},
        }
        for source, track in tracks.items():
            for mode in ("animation", "replacement"):
                with self.subTest(source=source, mode=mode):
                    with fake_sam3_unpack_masks(unpacked), redirect_stdout(io.StringIO()):
                        driving_mask, reference_mask = colored.build(
                            {**track, "orig_size": (16, 32), "n_frames": 5}, sort_by="none"
                        )
                    self.assertEqual((5, 16, 32, 3), tuple(driving_mask.shape))
                    self.assertEqual((1, 16, 32, 3), tuple(reference_mask.shape))
                    self.assertEqual(torch.float32, driving_mask.dtype)
                    self.assertTrue(torch.equal(reference_mask, torch.ones_like(reference_mask)))
                    if source != "empty":
                        self.assertTrue(torch.equal(driving_mask[..., :16, 2], torch.ones((5, 16, 16))))
                    self.assertEqual(0, torch.count_nonzero(driving_mask[..., 16:, :]).item())
                    before = driving_mask.clone()
                    pose = torch.full((5, 16, 32, 3), 0.25)
                    driving = torch.full((5, 16, 32, 3), 0.75)
                    condition, = condition_node.build(
                        pose_video_mask=driving_mask, ref_image=torch.zeros((1, 16, 32, 3)),
                        ref_mask=reference_mask, mode=mode, width=32, height=16, num_frames=5,
                        pose_video=pose, driving_video=driving,
                    )
                    self.assertIs(driving if mode == "replacement" else pose, condition.pose_video)
                    self.assertEqual(mode == "replacement", condition.replace_flag)
                    self.assertEqual((5, 16, 32), tuple(condition.driving_mask_indices.shape))
                    self.assertTrue(torch.equal(before, driving_mask), "mode normalization must not mutate RGB input")
                    expected_background = 0 if mode == "replacement" else -1
                    self.assertTrue((condition.driving_mask_indices[..., 16:] == expected_background).all())
                    # The no-mask reference fallback keeps the full reference latent in both modes.
                    self.assertTrue((condition.ref_mask_indices == 0).all())
                    self.assertTrue(torch.equal(reference_mask, torch.ones_like(reference_mask)))
                    payload, = adapter.build(condition)
                    self.assertIs(condition, payload["condition"])
                    self.assertEqual("WanVideoAddSCAIL2ConditionEmbeds", payload["target"]["native_consumer_node"])
                    self.assertEqual("scail_pose2.wanvideo_scail2_payload", payload["schema"]["name"])
                    self.assertEqual(1, payload["schema"]["version"])
                    runtime = payload["runtime_masks"]["driving"]
                    self.assertEqual((1, 2, 28, 1, 2), runtime.shape)
                    self.assertEqual((1, 1, 28, 2, 4), payload["runtime_masks"]["reference"].shape)
                    for phase in range(4):
                        self.assertEqual(1.0, payload["runtime_masks"]["reference"].value(
                            latent_frame=0, channel=7 * phase, row=1, col=3))
                    self.assertTrue(torch.isfinite(runtime.data).all())
                    for latent_frame in (0, 1):
                        for phase in range(4):
                            self.assertEqual(float(source != "empty"), runtime.value(
                                latent_frame=latent_frame, channel=7 * phase + 3, col=0))
                            self.assertEqual(float(mode == "replacement"), runtime.value(
                                latent_frame=latent_frame, channel=7 * phase, col=1))
                            self.assertEqual(float(mode == "replacement" and source == "empty"), runtime.value(
                                latent_frame=latent_frame, channel=7 * phase, col=0))
                    json.dumps(payload["schema"])


if __name__ == "__main__":
    unittest.main()
