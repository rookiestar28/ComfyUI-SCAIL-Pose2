# ComfyUI-SCAIL-Pose2

ComfyUI-SCAIL-Pose2 is a ComfyUI custom node package for SCAIL and SCAIL-2 pose and mask preprocessing. It prepares pose images, DWPose-compatible pose data, NLF pose renders, SCAIL-2 condition metadata, and explicit WanVideoWrapper-oriented adapter outputs. It is designed to work with ComfyUI-WanVideoWrapper as the downstream generation owner.

## Table of Contents

- [Installation](#installation)
- [Optional Dependencies](#optional-dependencies)
- [Node Groups](#node-groups)
  - [SCAIL-Pose](#scail-pose)
  - [WanAnimatePreprocess](#wananimatepreprocess)
  - [WanVideoWrapper](#wanvideowrapper)
  - [SCAIL-Pose2 / WanVideoWrapper](#scail-pose2--wanvideowrapper)
  - [SCAIL-Pose2 / SAM3](#scail-pose2--sam3)
  - [SCAIL-Pose2 / SCAIL-2](#scail-pose2--scail-2)
- [Native SCAIL-2 Workflow Notes](#native-scail-2-workflow-notes)
- [License](#license)

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

| Node | Main Inputs / Parameters | Outputs | Purpose / Notes |
| --- | --- | --- | --- |
| `NLFModelLoader` | `nlf_model`: NLF `.safetensors` model file from ComfyUI `models/nlf/`. | `nlf_model` (`NLF_MODEL`) | Loads this package's NLF model format for the core SCAIL-Pose flow. |
| `NLFPredictPoses` | `nlf_model`: output from `NLFModelLoader`.<br>`images`: input image/video frame batch.<br>`per_batch`: frames per inference batch; `-1` processes all at once, `1` uses less VRAM.<br>`num_aug`: test-time augmentations; higher is slower but can improve pose quality.<br>`detector_threshold`: person detection confidence threshold. | `pose_results` (`NLFPRED`), `bboxes` (`BBOX`) | Detects people and predicts NLF 3D pose data from image frames. |
| `ConvertOpenPoseKeypointsToDWPose` | `keypoints`: OpenPose-style keypoints.<br>`max_people`: maximum people to process per frame. | `dw_poses` (`DWPOSES`) | Converts OpenPose-compatible keypoints into the DWPose structure used by the render/alignment path. |

WanVideoWrapper also provides its own `(Download)Load NLF Model`, `Load NLF Model`, and `NLF Predict` nodes. Those wrapper nodes use the wrapper `NLFMODEL` socket and `.torchscript` model files. `NLF_MODEL` and `NLFMODEL` are not directly wire-compatible; use this package's NLF nodes for the core SCAIL-Pose flow, and use WanVideoWrapper's NLF nodes for wrapper-owned torchscript workflows.

### WanAnimatePreprocess

| Node | Main Inputs / Parameters | Outputs | Purpose / Notes |
| --- | --- | --- | --- |
| `PoseDetectionVitPoseToDWPose` | `vitpose_model`: loaded ViTPose detector/model bundle.<br>`images`: input image/video frame batch. | `dw_poses` (`DWPOSES`) | Runs the ViTPose detection path and converts detected pose metadata into DWPose-compatible data. |

### WanVideoWrapper

| Node | Main Inputs / Parameters | Outputs | Purpose / Notes |
| --- | --- | --- | --- |
| `RenderNLFPoses` | `nlf_poses`: NLF pose predictions.<br>`width`, `height`: render resolution.<br>`dw_poses`: optional 2D pose overlay/alignment data.<br>`ref_dw_pose`: optional reference pose for alignment.<br>`draw_face`, `draw_hands`: include face/hand keypoints.<br>`render_device`: Taichi device selection when using Taichi rendering.<br>`scale_hands`: hand scaling during reference alignment.<br>`render_backend`: `taichi` or `torch`; Taichi falls back to Torch when unavailable. | `image` (`IMAGE`), `mask` (`MASK`) | Renders pose images for pose-control workflows. `image` is the pose-rendered sequence; `mask` is the render alpha/visibility mask and is not a SCAIL-2 colored semantic mask. |
| `SaveNLFPosesAs3D` | `nlf_poses`: NLF pose predictions.<br>`filename_prefix`: output folder/file prefix.<br>`fps`: GLB animation frame rate.<br>`cylinder_radius`: bone-cylinder radius. | `output_path` (`STRING`) | Exports NLF pose data as a GLB animation for inspection, debugging, or external 3D workflow checks. |

### SCAIL-Pose2 / WanVideoWrapper

| Node | Main Inputs / Parameters | Outputs | Purpose / Notes |
| --- | --- | --- | --- |
| `SCAILPose2WanVideoSCAIL2Adapter` | `condition`: validated `SCAIL2_CONDITION` from `SCAILPose2SCAIL2Condition`.<br>`degrade_to_v1`: requests lossy v1 fallback metadata when supported.<br>`allow_degradation`: must be enabled together with `degrade_to_v1` before fallback data may be retained. | `condition` (`SCAIL2_WANVIDEO_PAYLOAD`) | Converts the validated SCAIL-2 condition into the versioned payload consumed by wrapper-native SCAIL-2 nodes. The UI output is named `condition`, but its ComfyUI type is `SCAIL2_WANVIDEO_PAYLOAD`, not the original `SCAIL2_CONDITION`. |

Native WanVideoWrapper wiring uses `WanVideoAddSCAIL2ConditionEmbeds`: connect this adapter node's `condition` output to that wrapper node's `condition` input, then connect the wrapper node's `image_embeds` output to `WanVideo Sampler v2.image_embeds`. The wrapper native path stores SCAIL-2 data under `scail2_embeds` and rejects simultaneous legacy `scail_embeds`.

The older standalone v1 image adapter public node is no longer registered. Its validation behavior is now used internally by the SCAIL-2 adapter fallback path.

### SCAIL-Pose2 / SAM3

| Node | Main Inputs / Parameters | Outputs | Purpose / Notes |
| --- | --- | --- | --- |
| `SCAIL2SAM3DependencyCheck` | No inputs. | `status` (`STRING`) | Checks whether optional SAM3 preprocessing dependencies are available in the active ComfyUI environment without making SAM3 a startup requirement. |
| `SCAILPose2ColoredMask` | `driving_track_data`: SAM3 track data from ComfyUI's built-in `SAM3 Video Track.track_data` for the driving video.<br>`object_indices`: optional comma-separated object indices to keep; blank keeps all tracked objects.<br>`sort_by`: identity ordering strategy: `left_to_right`, `area`, or `none`.<br>`ref_track_data`: optional SAM3 track data for reference identities; do not connect together with `ref_mask`.<br>`ref_mask`: optional plain reference subject `MASK`; do not connect together with `ref_track_data`. | `pose_video_mask` (`IMAGE`), `reference_image_mask` (`IMAGE`) | Renders identity-colored SCAIL-2 RGB semantic masks from SAM3 tracks. Packed SAM3 masks are unpacked and resized to `orig_size`. The preview normally shows colored subjects on black background; it is semantic mask data, not the original video. |

### SCAIL-Pose2 / SCAIL-2

| Node | Main Inputs / Parameters | Outputs | Purpose / Notes |
| --- | --- | --- | --- |
| `SCAILPose2SCAIL2Condition` | `pose_video`: pose-rendered image sequence, normally from `RenderNLFPoses.image`; not the raw driving video.<br>`pose_video_mask`: SCAIL-2 colored semantic mask sequence, normally from `SCAILPose2ColoredMask.pose_video_mask`.<br>`ref_image`: reference subject image.<br>`ref_mask`: SCAIL-2 reference semantic mask, normally `SCAILPose2ColoredMask.reference_image_mask` when no separate mask is supplied.<br>`mode`: `animation` or `replacement`.<br>`width`, `height`: final generation dimensions.<br>`num_frames`: expected pose/mask frame count.<br>`additional_ref_image`, `additional_ref_mask`: optional paired extra reference inputs. | `condition` (`SCAIL2_CONDITION`) | Builds the validated SCAIL-2 condition payload. `pose_video`, `pose_video_mask`, `width`, `height`, and `num_frames` must align. Long-video context length and overlap are controlled downstream by WanVideoWrapper context options. |
| `SCAILPose2ReplacementDenoiseMask` | `condition`: validated `SCAIL2_CONDITION`.<br>`pose_video_mask`: same raw colored semantic mask used by the Condition node.<br>`grow_pixels`: expands the subject denoise area before sampler use.<br>`blur_pixels`: softens the denoise mask edge. | `mask` (`MASK`), `summary` (`STRING`) | Builds the `WanVideoEncode.mask` input for replacement/background-lock workflows. In `replacement` mode, subject pixels are `1.0` and background preserve pixels are `0.0`. In `animation` mode, it emits an all-`1.0` passthrough mask with SCAIL-Pose2 metadata that disables `WanVideoEncode.samples`, preventing raw driving-video influence in standard SCAIL-Pose2 workflows. |

## Native SCAIL-2 Workflow Notes

For native WanVideoWrapper SCAIL-2 workflows, connect `SCAILPose2WanVideoSCAIL2Adapter.condition` to `WanVideoAddSCAIL2ConditionEmbeds.condition`, then send that node's `image_embeds` output to `WanVideo Sampler v2.image_embeds`. Use `WanVideo Context Options` for long-video `context_frames`, stride, and overlap by connecting its `context_options` output to `WanVideo Sampler v2.context_options`.

When using SAM3 masks, connect `SCAILPose2ColoredMask.pose_video_mask` to `SCAILPose2SCAIL2Condition.pose_video_mask`. If you do not provide a separate reference mask, connect `SCAILPose2ColoredMask.reference_image_mask` to `SCAILPose2SCAIL2Condition.ref_mask` so the Condition payload still receives the reference-mask signal expected by SCAIL-2 conditioning.

Keep `SCAILPose2SCAIL2Condition.width` and `height` at the final generation size. Do not halve them to match pose latents. The WanVideo adapter packs reference masks at full latent size and packs driving masks at pose-control latent size so wrapper pose latents and driving masks share temporal/spatial shape.

Replacement mode has two separate data paths:

1. SCAIL-2 conditioning: `SCAILPose2WanVideoSCAIL2Adapter.condition` -> `WanVideoAddSCAIL2ConditionEmbeds.condition` -> `WanVideo Sampler v2.image_embeds`.
2. Hard background preservation: original driving video -> `WanVideoEncode.driving_video`, `SCAILPose2ReplacementDenoiseMask.mask` -> `WanVideoEncode.mask`, then `WanVideoEncode.samples` -> `WanVideo Sampler v2.samples`.

Use both paths for replacement background lock. SCAIL-2 conditioning guides the generated subject and reference behavior, but it does not hard-freeze the original driving-video background by itself. In WanVideoWrapper, enable `add_noise_to_samples` on `WanVideo Sampler v2` for clean-video replacement/inpaint workflows.

When the Condition node is set to `animation`, the replacement denoise mask path is automatically neutralized by emitting an all-`1.0` mask with SCAIL-Pose2 metadata that disables `WanVideoEncode.samples`. This keeps an already-wired workflow executable without letting the raw driving video influence sampler initialization. If a workflow bypasses `SCAILPose2ReplacementDenoiseMask` and feeds arbitrary latents directly into `WanVideo Sampler v2.samples`, that is treated as an explicit generic vid2vid path outside this SCAIL-Pose2 mode gate.

The Colored Mask preview normally shows a black background with colored subject regions. That is expected: the preview is a semantic mask, not the original video. The denoise mask derives subject pixels from this raw semantic mask before any Condition-node mode-specific semantic conversion.

Reference identity depends on the full reference path being wired correctly: connect `ref_image`, connect `ref_mask` or `reference_image_mask`, and keep the wrapper CLIP/reference embed path connected through the SCAIL-2 condition embeds node. If reference identity still drifts, check the reference mask quality, CLIP/reference image connection, and replacement denoise mask margin controls (`grow_pixels` and `blur_pixels`).

The Colored Mask, Condition, and WanVideo adapter nodes emit safe progress/log summaries for long work. These summaries report metadata such as shape, dtype, device, frame count, object count, and elapsed time; they do not log raw mask pixels, prompts, model paths, or media contents.

The SCAIL-Pose2 Condition node does not expose SCAIL-2 segment or continuation controls. Wrapper context windows are supported by the wrapper-native SCAIL-2 path, but official SCAIL-2 clean-history continuation is not claimed by this package.

## License

MIT License
