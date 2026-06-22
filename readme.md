# ComfyUI-SCAIL-Pose2

ComfyUI-SCAIL-Pose2 is a ComfyUI custom node package for SCAIL-2 pose and mask preprocessing. The public workflow documented here focuses on this repo's SCAIL-Pose2 nodes: SAM3 colored masks, SCAIL-2 condition payloads, WanVideoWrapper adapter payloads, replacement denoise masks, and legacy condition-video utilities.

## Table of Contents

- [Installation](#installation)
- [Optional Dependencies](#optional-dependencies)
- [Node Groups](#node-groups)
  - [SCAIL-Pose2 / Adapter](#scail-pose2--adapter)
  - [SCAIL-Pose2 / SAM3](#scail-pose2--sam3)
  - [SCAIL-Pose2 / SCAIL-2](#scail-pose2--scail-2)
- [Native SCAIL-2 Workflow Notes](#native-scail-2-workflow-notes)
  - [Data Contract](#data-contract)
  - [Pose Geometry](#pose-geometry)
  - [Replacement Mode](#replacement-mode)
    - [Required Wiring](#required-wiring)
    - [Reference And Shape Tuning](#reference-and-shape-tuning)
    - [Preview Behavior](#preview-behavior)
  - [Notes](#notes)
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

- SAM3 preprocessing is optional. SAM3-specific imports are attempted only when the SAM3 execution path runs; missing dependency errors should point to the optional SAM3 requirement instead of breaking package import.
- ComfyUI-WanVideoWrapper is the intended downstream generation partner, but this package does not import it at startup.

## Node Groups

### SCAIL-Pose2 / Adapter

| Node | Main Inputs / Parameters | Outputs | Purpose / Notes |
| --- | --- | --- | --- |
| `SCAILPose2WanVideoSCAIL2Adapter` | `condition`: validated `SCAIL2_CONDITION` from `SCAILPose2SCAIL2Condition`.<br>`degrade_to_v1`: requests lossy v1 fallback metadata when supported.<br>`allow_degradation`: must be enabled together with `degrade_to_v1` before fallback data may be retained. | `condition` (`SCAIL2_WANVIDEO_PAYLOAD`) | Converts the validated SCAIL-2 condition into this repo's versioned downstream adapter payload. The UI output is named `condition`, but its ComfyUI type is `SCAIL2_WANVIDEO_PAYLOAD`, not the original `SCAIL2_CONDITION`. |

The older standalone v1 image adapter public node is no longer registered. Its validation behavior is used internally by the SCAIL-2 adapter fallback path.

### SCAIL-Pose2 / SAM3

| Node | Main Inputs / Parameters | Outputs | Purpose / Notes |
| --- | --- | --- | --- |
| `SCAIL2SAM3DependencyCheck` | No inputs. | `status` (`STRING`) | Checks whether optional SAM3 preprocessing dependencies are available in the active ComfyUI environment without making SAM3 a startup requirement. |
| `SCAILPose2ColoredMask` | `driving_track_data`: SAM3 track data for the driving video.<br>`object_indices`: optional comma-separated object indices to keep; blank keeps all tracked objects.<br>`sort_by`: identity ordering strategy: `left_to_right`, `area`, or `none`.<br>`ref_track_data`: optional SAM3 track data for reference identities; do not connect together with `ref_mask`.<br>`ref_mask`: optional plain reference subject `MASK`; do not connect together with `ref_track_data`. | `pose_video_mask` (`IMAGE`), `reference_image_mask` (`IMAGE`) | Renders identity-colored SCAIL-2 RGB semantic masks from SAM3 tracks. Packed SAM3 masks are unpacked and resized to `orig_size`. The preview normally shows colored subjects on black background; it is semantic mask data, not the original video. |

### SCAIL-Pose2 / SCAIL-2

| Node | Main Inputs / Parameters | Outputs | Purpose / Notes |
| --- | --- | --- | --- |
| `SCAILPose2SCAIL2Condition` | `pose_video`: animation-mode rendered pose image sequence.<br>`driving_video`: replacement-mode original driving video sequence.<br>`pose_video_mask`: SCAIL-2 colored semantic mask sequence, normally from `SCAILPose2ColoredMask.pose_video_mask`.<br>`ref_image`: reference subject image.<br>`ref_mask`: SCAIL-2 reference semantic mask, normally `SCAILPose2ColoredMask.reference_image_mask` when no separate mask is supplied.<br>`mode`: `animation` or `replacement`.<br>`width`, `height`: final generation dimensions.<br>`num_frames`: expected pose/mask frame count.<br>`additional_ref_image`, `additional_ref_mask`: optional paired extra reference inputs. | `condition` (`SCAIL2_CONDITION`) | Builds the validated SCAIL-2 condition payload. Both `pose_video` and `driving_video` can stay wired; `mode` selects the active source. The active source, `pose_video_mask`, `width`, `height`, and `num_frames` must share the intended final canvas and frame count. Long-video context length and overlap are downstream responsibilities. |
| `SCAILPose2PoseMaskGeometryAlign` | `pose_video`: rendered pose image sequence, usually from `RenderNLFPoses.image`.<br>`pose_video_mask`: driving semantic mask sequence from `SCAILPose2ColoredMask.pose_video_mask`. | `pose_video` (`IMAGE`), `summary` (`STRING`) | Scales and translates rendered pose foregrounds so their bbox matches the SAM3-derived driving mask bbox. It infers geometry from the actual image sizes. Use this for already-rendered pose images or manual repair workflows. |
| `SCAILPose2ReplacementConditionVideo` | `driving_video`: original replacement-mode driving video sequence.<br>`pose_video_mask`: same raw colored semantic mask used by the Condition node.<br>`mask_preset`: `custom`, `tight`, `default`, `loose`, or `soft`; non-custom presets override grow/blur values.<br>`grow_pixels`, `blur_pixels`: custom subject-region expansion and edge softness.<br>`suppression_mode`: `blur_fill`, `mean_fill`, `black_fill`, `white_fill`, or `noise_fill`.<br>`suppression_strength`: `0.0` keeps the original subject; `1.0` fully suppresses subject pixels inside the mask.<br>`noise_seed`: deterministic seed for `noise_fill`. | `driving_video_condition` (`IMAGE`), `summary` (`STRING`) | Legacy/experimental/manual fallback that suppresses original subject pixels before condition encoding. It is not the canonical replacement route because altering this video can weaken the official SCAIL-2 pose-latent motion signal. Prefer raw `driving_video` plus the sampler denoise-mask path. |
| `SCAILPose2ReplacementDenoiseMask` | `condition`: validated `SCAIL2_CONDITION`.<br>`pose_video_mask`: same raw colored semantic mask used by the Condition node.<br>`mask_preset`: `custom`, `tight`, `default`, `loose`, or `soft`; non-custom presets override grow/blur values.<br>`grow_pixels`: expands the subject denoise area before sampler use.<br>`blur_pixels`: softens the denoise mask edge. | `mask` (`MASK`), `summary` (`STRING`) | Builds a standard ComfyUI `MASK` for replacement/background-lock workflows. In `replacement` mode, subject pixels are `1.0` and background preserve pixels are `0.0`. In `animation` mode, it emits an all-`1.0` passthrough mask with SCAIL-Pose2 metadata that disables this repo's background-lock samples path in compatible downstream integrations. The summary includes mask coverage and margin diagnostics. |

## Native SCAIL-2 Workflow Notes

### Data Contract

This README documents this repo's SCAIL-Pose2 nodes only. Downstream wrapper nodes are not listed as repo nodes in `Node Groups`.

Use `SAM3 Video Track.track_data` as the SAM3 source for `SCAILPose2ColoredMask.driving_track_data`. Use `SCAILPose2ColoredMask.pose_video_mask` as `SCAILPose2SCAIL2Condition.pose_video_mask`. Colored Mask `ref_mask` is optional; if no separate reference mask is available, use `SCAILPose2ColoredMask.reference_image_mask` as `SCAILPose2SCAIL2Condition.ref_mask`.

Keep `SCAILPose2SCAIL2Condition.width` and `height` at the final generation size. Do not halve them to match pose latents; downstream integrations may encode pose inputs at the pose-control latent size internally. The adapter handles the required reference-mask and driving-mask packing internally.

`SCAILPose2WanVideoSCAIL2Adapter.condition` is the SCAIL-2 adapter payload intended for compatible downstream SCAIL-2 embedding consumers such as `WanVideoAddSCAIL2ConditionEmbeds`. Context windows and overlap remain downstream generation-wrapper responsibilities, commonly configured through WanVideo Context Options. SCAIL-2 clean-history continuation is not claimed by this repo.

The adapter can report lossy v1 fallback metadata only when degradation is explicitly requested and allowed. Treat that path as compatibility metadata, not full SCAIL-2 parity.

### Pose Geometry

Animation mode uses rendered pose images. Those images may be half the final generation resolution, but they must be a coordinate-equivalent downsample of the final canvas. Half resolution does not mean a separate crop, a different projection, or reference-pose rescaling that changes the driving subject bbox.

Replacement mode does not use rendered NLF skeletons as the canonical condition video. Connect the original `driving_video` sequence as raw `driving_video` directly to `SCAILPose2SCAIL2Condition.driving_video` and use `SCAILPose2ColoredMask.pose_video_mask` as the semantic replacement mask. `SCAILPose2PoseMaskGeometryAlign` remains a manual repair utility for already-rendered pose images, not a required replacement-mode step.

### Replacement Mode

#### Required Wiring

Replacement workflows use these repo outputs:

1. `SCAILPose2WanVideoSCAIL2Adapter.condition` for SCAIL-2 conditioning.
2. `SCAILPose2ReplacementDenoiseMask.mask` for hard background preservation.

SCAIL-2 conditioning guides subject/reference behavior; it does not hard-freeze the original background by itself. Replacement background lock also requires the downstream video encode samples path to receive the original `driving_video` and this repo's replacement denoise mask, then pass those samples into the sampler. In `animation` mode, `SCAILPose2ReplacementDenoiseMask` emits an all-`1.0` passthrough mask with metadata that disables compatible downstream background-lock samples.

For replacement, `SCAILPose2SCAIL2Condition.driving_video` should receive the raw `driving_video` directly. Do not route `RenderNLFPoses` into that input for replacement mode; rendered pose skeletons cannot preserve the original subject proportions relative to the SAM3 mask. If both `pose_video` and `driving_video` are connected, the Condition node automatically uses `pose_video` for `animation` mode and `driving_video` for `replacement` mode.

When original subject body shape leaks into replacement output, first verify the downstream samples/noise-mask path: original `driving_video` must be encoded with `SCAILPose2ReplacementDenoiseMask.mask`, then those samples must reach the sampler. `SCAILPose2ReplacementConditionVideo` is retained only as a legacy/experimental/manual fallback for unusual experiments, and using it may reduce pose accuracy because it changes the video that becomes SCAIL-2 pose latents.

Compatible downstream integrations should preserve the replacement mask's SCAIL-Pose2 metadata so subject pixels remain `1.0` replace/denoise areas and background pixels remain `0.0` preserve areas after latent conversion.

#### Reference And Shape Tuning

The canonical route keeps the replacement condition video raw so the official SCAIL-2 pose-latent signal remains intact. If you deliberately use the legacy `SCAILPose2ReplacementConditionVideo` fallback, treat it as an experiment after confirming the sampler mask path is correct; a conservative starting point is `mask_preset=custom`, `grow_pixels=8`, `blur_pixels=0`, `suppression_mode=blur_fill`, and `suppression_strength=1.0`.

Compatible WanVideoWrapper builds may expose SCAIL-2 strength controls on their SCAIL-2 condition-embeds node: `ref_image_strength`, `ref_mask_strength`, `condition_video_strength`, and `driving_mask_strength`. Their default `1.0` values preserve previous behavior. To bias replacement toward the reference image, raise `ref_image_strength` carefully and reduce `condition_video_strength` only when original body shape still dominates.

#### Preview Behavior

Early sampler previews can still show a noisy or incomplete original background even when the background-lock path is wired correctly. During early denoise steps, the preserved background latent is also highly noised; judge background preservation from later previews or the final decoded output, not from the first preview frames alone.

### Notes

- Colored Mask previews show colored subjects on black background because they are semantic masks, not original video frames.
- Reference identity depends on `ref_image`, `ref_mask` / `reference_image_mask`, and the downstream reference embedding path.
- Long-running nodes log safe progress/log summaries such as shape, dtype, frame count, object count, and elapsed time.
- The Condition node does not expose SCAIL-2 segment or continuation controls; downstream context windows are owned by the generation wrapper.
- This repo's restored NLF pose loader returns `NLF_MODEL`; the WanVideoWrapper NLF family may expose `NLFMODEL` through nodes such as `(Download)Load NLF Model`. Those model contracts are not directly wire-compatible: this repo expects local NLF `.safetensors` assets, while wrapper-side NLF loaders may manage `.torchscript` assets and downloads.

## License

MIT License
