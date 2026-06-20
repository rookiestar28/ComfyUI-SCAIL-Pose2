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

- `NLFModelLoader`: loads an NLF pose model from the ComfyUI `models/nlf/` folder and returns an `NLF_MODEL` handle for pose prediction. This core loader expects the package's `.safetensors` NLF model format.
- `NLFPredictPoses`: runs the loaded NLF model on input images and returns `NLFPRED` pose data plus detected bounding boxes.
- `ConvertOpenPoseKeypointsToDWPose`: converts OpenPose-style keypoint data into the DWPose-compatible structure used by the render and alignment path.

WanVideoWrapper also provides its own `(Download)Load NLF Model`, `Load NLF Model`, and `NLF Predict` nodes. Those wrapper nodes use the wrapper `NLFMODEL` socket and `.torchscript` model files. `NLF_MODEL` and `NLFMODEL` are not directly wire-compatible; use this package's NLF nodes for the core SCAIL-Pose flow, and use WanVideoWrapper's NLF nodes for wrapper-owned torchscript workflows.

### WanAnimatePreprocess

- `PoseDetectionVitPoseToDWPose`: runs the ViTPose detection path and converts detected pose metadata into the DWPose-compatible format expected by downstream preprocessing.

### WanVideoWrapper

- `RenderNLFPoses`: renders `NLFPRED` pose data into pose images and masks for WanVideoWrapper-oriented pose-control workflows.
- `SaveNLFPosesAs3D`: exports NLF pose data as a GLB animation for inspection, debugging, or external 3D workflow checks.

### SCAIL-Pose2 / WanVideoWrapper

- `SCAILPose2WanVideoSCAIL2Adapter`: converts a validated `SCAIL2_CONDITION` into a versioned SCAIL-2 adapter payload. Its only output is named `condition` in the UI, but its ComfyUI type is `SCAIL2_WANVIDEO_PAYLOAD`; it is an adapter condition payload, not the original `SCAIL2_CONDITION` object.
- Native WanVideoWrapper wiring uses `WanVideoAddSCAIL2ConditionEmbeds`: connect this adapter node's `condition` output to that wrapper node's `condition` input, then connect the wrapper node's `image_embeds` output to `WanVideo Sampler v2.image_embeds`. The wrapper native path stores SCAIL-2 data under `scail2_embeds` and rejects simultaneous legacy `scail_embeds`.
- When `degrade_to_v1` and `allow_degradation` are both enabled, lossy v1 fallback data may be retained inside the adapter payload for internal compatibility checks, but the public ComfyUI node no longer exposes separate v1 image or dimension output sockets.
- The older standalone v1 image adapter public node is no longer registered. Its validation behavior is now used internally by the SCAIL-2 adapter fallback path.

### SCAIL-Pose2 / SAM3

- `SCAIL2SAM3DependencyCheck`: checks whether optional SAM3 preprocessing dependencies are available in the active ComfyUI environment without making SAM3 a startup requirement.
- `SCAILPose2ColoredMask`: renders identity-colored SCAIL-2 RGB masks from existing SAM3 track data. Connect ComfyUI's built-in `SAM3 Video Track.track_data` output to this node's `driving_track_data` input. It supports shared identity sorting, object filtering, animation/replacement background semantics, optional reference track data, and optional plain reference masks. Colored Mask `ref_mask` is optional; if no reference track or plain mask is connected, the node emits a solid `reference_image_mask`. Packed SAM3 masks are unpacked and resized to `orig_size` before rendering.

### SCAIL-Pose2 / SCAIL-2

- `SCAILPose2SCAIL2Condition`: builds a validated `SCAIL2_CONDITION` payload from `pose_video`, `pose_video_mask`, `ref_image`, `ref_mask`, optional `additional_ref_image` / `additional_ref_mask`, mode, dimensions, and `num_frames`. Supported modes are `animation` and `replacement`; pose-rendered driving videos are animation-mode inputs. Long-video context length and overlap should be controlled by WanVideoWrapper context options downstream.
- `SCAILPose2ReplacementDenoiseMask`: builds a standard ComfyUI `MASK` for replacement workflows. Connect the same raw `pose_video_mask` used by the Condition node plus the validated `SCAIL2_CONDITION`; the output mask uses `1.0` for the subject/replace area and `0.0` for the original background/preserve area.

## Native SCAIL-2 Workflow Notes

For native WanVideoWrapper SCAIL-2 workflows, connect `SCAILPose2WanVideoSCAIL2Adapter.condition` to `WanVideoAddSCAIL2ConditionEmbeds.condition`, then send that node's `image_embeds` output to `WanVideo Sampler v2.image_embeds`. Use `WanVideo Context Options` for long-video `context_frames`, stride, and overlap by connecting its `context_options` output to `WanVideo Sampler v2.context_options`.

When using SAM3 masks, connect `SCAILPose2ColoredMask.pose_video_mask` to `SCAILPose2SCAIL2Condition.pose_video_mask`. If you do not provide a separate reference mask, connect `SCAILPose2ColoredMask.reference_image_mask` to `SCAILPose2SCAIL2Condition.ref_mask` so the Condition payload still receives the reference-mask signal expected by SCAIL-2 conditioning.

Keep `SCAILPose2SCAIL2Condition.width` and `height` at the final generation size. Do not halve them to match pose latents. The WanVideo adapter packs reference masks at full latent size and packs driving masks at pose-control latent size so wrapper pose latents and driving masks share temporal/spatial shape.

Replacement mode has two separate data paths:

1. SCAIL-2 conditioning: `SCAILPose2WanVideoSCAIL2Adapter.condition` -> `WanVideoAddSCAIL2ConditionEmbeds.condition` -> `WanVideo Sampler v2.image_embeds`.
2. Hard background preservation: original driving video -> `WanVideoEncode.image`, `SCAILPose2ReplacementDenoiseMask.mask` -> `WanVideoEncode.mask`, then `WanVideoEncode.samples` -> `WanVideo Sampler v2.samples`.

Use both paths for replacement background lock. SCAIL-2 conditioning guides the generated subject and reference behavior, but it does not hard-freeze the original driving-video background by itself. In WanVideoWrapper, enable `add_noise_to_samples` on `WanVideo Sampler v2` for clean-video replacement/inpaint workflows.

The Colored Mask preview normally shows a black background with colored subject regions. That is expected: the preview is a semantic mask, not the original video. For replacement workflows, the new denoise mask derives subject pixels from this raw semantic mask before the Condition node's replacement-mode semantic conversion.

Reference identity depends on the full reference path being wired correctly: connect `ref_image`, connect `ref_mask` or `reference_image_mask`, and keep the wrapper CLIP/reference embed path connected through the SCAIL-2 condition embeds node. If reference identity still drifts, check the reference mask quality, CLIP/reference image connection, and replacement denoise mask margin controls (`grow_pixels` and `blur_pixels`).

The Colored Mask, Condition, and WanVideo adapter nodes emit safe progress/log summaries for long work. These summaries report metadata such as shape, dtype, device, frame count, object count, and elapsed time; they do not log raw mask pixels, prompts, model paths, or media contents.

The SCAIL-Pose2 Condition node does not expose SCAIL-2 segment or continuation controls. Wrapper context windows are supported by the wrapper-native SCAIL-2 path, but official SCAIL-2 clean-history continuation is not claimed by this package.

## License

MIT License
