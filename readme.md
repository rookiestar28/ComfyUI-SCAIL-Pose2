# ComfyUI-SCAIL-Pose2

ComfyUI-SCAIL-Pose2 is a ComfyUI custom node package for SCAIL and SCAIL-2 pose and mask preprocessing. It prepares pose images, DWPose-compatible pose data, NLF pose renders, SCAIL-2 condition metadata, and explicit WanVideoWrapper-oriented adapter outputs. It is designed to work with ComfyUI-WanVideoWrapper as the downstream generation owner.

## Installation

Recommended: install this package with ComfyUI-Manager by searching for `ComfyUI-SCAIL-Pose2`, installing it, and restarting ComfyUI.

Manual installation is also supported. Clone this package into the ComfyUI custom nodes directory, then install the lightweight package requirements into the same Python environment used by ComfyUI.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/rookiestar28/ComfyUI-SCAIL-Pose2.git
cd ComfyUI-SCAIL-Pose2
python -m pip install -r requirements.txt
```

If you use the Windows portable ComfyUI build, run the install command with the portable Python executable instead of a global Python.

## Optional Dependencies

ComfyUI provides the core runtime, including Torch in normal installations. This package keeps heavy or workflow-specific components lazy so ComfyUI can start even when optional stacks are absent.

- Taichi is optional. If Taichi rendering is selected but unavailable, NLF rendering falls back to the Torch renderer.
- SAM3 preprocessing is optional. SAM3-specific imports are attempted only when the SAM3 execution path runs; missing dependency errors should point to the optional SAM3 requirement instead of breaking package import.
- ComfyUI-WanVideoWrapper is the intended downstream generation partner, but this package does not import it at startup.

## Node Groups

### SCAIL-Pose

- `NLFModelLoader`: loads an NLF pose model from the ComfyUI `models/nlf/` folder and returns an `NLF_MODEL` handle for pose prediction.
- `NLFPredictPoses`: runs the loaded NLF model on input images and returns `NLFPRED` pose data plus detected bounding boxes.
- `ConvertOpenPoseKeypointsToDWPose`: converts OpenPose-style keypoint data into the DWPose-compatible structure used by the render and alignment path.

### WanAnimatePreprocess

- `PoseDetectionVitPoseToDWPose`: runs the ViTPose detection path and converts detected pose metadata into the DWPose-compatible format expected by downstream preprocessing.

### WanVideoWrapper

- `RenderNLFPoses`: renders `NLFPRED` pose data into pose images and masks for WanVideoWrapper-oriented pose-control workflows.
- `SaveNLFPosesAs3D`: exports NLF pose data as a GLB animation for inspection, debugging, or external 3D workflow checks.

### SCAIL-Pose2 / WanVideoWrapper

- `SCAILPose2WanVideoSCAIL2Adapter`: converts a validated `SCAIL2_CONDITION` into a versioned SCAIL-2 adapter payload. The first output is named `condition` in the UI, but its ComfyUI type is `SCAIL2_WANVIDEO_PAYLOAD`; it is an adapter condition payload, not the original `SCAIL2_CONDITION` object.
- Native WanVideoWrapper wiring uses `WanVideoAddSCAIL2ConditionEmbeds`: connect this adapter node's `condition` output to that wrapper node's `condition` input, then connect the wrapper node's `image_embeds` output to `WanVideo Sampler v2.image_embeds`. The wrapper native path stores SCAIL-2 data under `scail2_embeds` and rejects simultaneous legacy `scail_embeds`.
- When `degrade_to_v1` and `allow_degradation` are both enabled, `SCAILPose2WanVideoSCAIL2Adapter` also exposes a lossy v1 fallback: `ref_image`, `pose_images`, `clip_ref_image`, `width`, `height`, and `num_frames`. These can be wired to WanVideoWrapper v1 SCAIL image nodes only when users accept the documented semantic losses.
- The older standalone v1 image adapter public node is no longer registered. Its validation behavior is now used internally by the SCAIL-2 adapter fallback path.

### SCAIL-Pose2 / SAM3

- `SCAIL2SAM3DependencyCheck`: checks whether optional SAM3 preprocessing dependencies are available in the active ComfyUI environment without making SAM3 a startup requirement.
- `SCAILPose2ColoredMask`: renders identity-colored SCAIL-2 RGB masks from existing SAM3 track data. It supports shared identity sorting, object filtering, animation/replacement background semantics, optional reference track data, and optional plain reference masks.

### SCAIL-Pose2 / SCAIL-2

- `SCAILPose2SCAIL2Condition`: builds a validated `SCAIL2_CONDITION` payload from `pose_video`, `pose_video_mask`, `ref_image`, `ref_mask`, optional `additional_ref_image` / `additional_ref_mask`, mode, dimensions, and `num_frames`. Supported modes are `animation` and `replacement`; pose-rendered driving videos are animation-mode inputs. Long-video context length and overlap should be controlled by WanVideoWrapper context options downstream.

## License

MIT License
