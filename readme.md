# ComfyUI-SCAIL-Pose2

ComfyUI-SCAIL-Pose2 is a ComfyUI custom node package for SCAIL-2 pose and mask preprocessing. The public workflow documented here focuses on this repo's SCAIL-Pose2 nodes: SAM3 colored masks, SCAIL-2 condition payloads, WanVideoWrapper adapter payloads, replacement denoise masks, and legacy condition-video utilities.

## Last Update

Version **1.1.5**:

- Added explicit historical and refreshed ComfyUI core source profiles. Host-contract verification
  preserves both revisions and checks each contract against its own committed source, including
  socket order and output names/types, with strict provenance and containment checks.
- Strengthened canonical template publication checks to reject conflicting, matching, empty, or
  named-only `widgets_values_named` state while preserving VHS-owned widget objects.
- Added registered Colored Mask -> Condition -> Adapter integration coverage for packed, regular,
  and empty SAM3 tracks in both animation and replacement modes, including reference fallback,
  input immutability, and final 28-channel mask payload values and shapes.
- Added recommended and compact SCAIL-2 replacement workflow templates with public-safe asset
  placeholders and stable KJNodes Set/Get pairing.
- Made native SCAIL-2 animation and replacement workflows use RGB conditioning directly without a
  mandatory NLF or skeleton-render branch.
- Set the replacement templates to the raw `driving_video` and colored semantic-mask route with
  independent empty embeds and no encoded sampler `samples`, avoiding hard subject/background
  splices by default.
- Added source-derived WanVideo host-contract validation and clean-clone workflow fixtures that
  catch socket, link, output-cardinality, host-family, and ignored-file drift.
- Repaired native animation, replacement, legacy SCAIL-v1, and WanAnimate fallback workflow
  contracts while retaining pose nodes only where the legacy host contract requires them.
- Routed the 81-frame context window through WanVideo Context Options and Sampler Extra Args before
  WanVideo Sampler V2 so the default templates do not serialize an invalid zero-sized context.

## Table of Contents

- [Last Update](#last-update)
- [Installation](#installation)
- [Optional Dependencies](#optional-dependencies)
- [Default Workflow Templates](#default-workflow-templates)
- [Node Groups](#node-groups)
  - [SCAIL-Pose2 / Adapter](#scail-pose2--adapter)
  - [SCAIL-Pose2 / SAM3](#scail-pose2--sam3)
  - [SCAIL-Pose2 / SCAIL-2](#scail-pose2--scail-2)
- [Native SCAIL-2 Workflow Notes](#native-scail-2-workflow-notes)
  - [Data Contract](#data-contract)
  - [Multi-Person Identity](#multi-person-identity)
  - [Pose Geometry](#pose-geometry)
  - [Replacement Mode](#replacement-mode)
    - [Required Wiring](#required-wiring)
    - [Reference And Shape Tuning](#reference-and-shape-tuning)
    - [Preview Behavior](#preview-behavior)
    - [Troubleshooting](#troubleshooting)
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

## Default Workflow Templates

ComfyUI discovers this package's loadable templates from `example_workflows/` in the Template
Browser:

- `example_workflows/scail2_replacement_default.json` is the recommended setup. It retains bounded
  reference, semantic-mask, comparison, and result preview nodes.
- `example_workflows/scail2_replacement_wanvideowrapper_compact.json` retains the final-output
  dependency graph and the same native WanVideoWrapper generation/control lane with fewer preview
  surfaces. The compact variant has not been independently live-smoke-tested; it is a static,
  contract-tested derivative of the recommended template. Both distributed JSON templates are
  statically validated and have not been independently live-smoke-tested as published.

Install or provide compatible versions of ComfyUI-SCAIL-Pose2, ComfyUI-WanVideoWrapper with
`WanVideoAddSCAIL2ConditionEmbeds`, ComfyUI-VideoHelperSuite, ComfyUI-KJNodes,
ComfyUI_Text_Processor, and a ComfyUI build that provides `SAM3_VideoTrack` and the required core
loaders. The compact template is WanVideoWrapper-focused in its generation/control lane; the full
workflow also needs SCAIL-Pose2 condition nodes, SAM3 tracking, utility/resize nodes, and video I/O.

The distributed templates use positional widget values. Their publication checks reject extra
`widgets_values_named` state so mode, context, assets, prompts, and LoRA settings have one authority.
VHS retains its extension-owned object-shaped `widgets_values`. User-saved workflows continue to
follow the active frontend's serialization settings. Source-contract checks and automated node
tests provide repository-side evidence; installed-host browser behavior and model inference need
separate validation with the actual core, frontend, wrapper, and other node packs in use.

After loading either template, Select every `select_*` placeholder for the driving video, reference
image, SCAIL-2 diffusion model, VAE, CLIP Vision model, text encoder, and SAM3 checkpoint. LoRA slots
are disabled by default. Replace the generic positive and negative prompt text, then verify the
generated width, height, frame count, and fixed context value before queuing.

Keep the paired KJNodes Set/Get variable name `driving_video` unchanged on both nodes. That name is
the virtual-link key, and the replacement Condition intentionally receives it through
`driving_video`; `pose_video` is the separate animation-mode input. Renaming only one end breaks the
virtual route, while repointing replacement mode to `pose_video` violates the mode contract.

The accepted no-samples route is the default: raw driving video and colored semantic masks feed the
SCAIL-2 condition, `WanVideoEmptyEmbeds` supplies independent shape/frame authority, and the sampler `samples` input is intentionally disconnected. Adding an encoded source-video samples/noise-mask
path is an optional experiment, not required wiring. That background hard splice can preserve a
different spatial boundary than the generated subject and visibly shift, clip, or stair-step the
scene at their intersection.

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
| `SCAILPose2ColoredMask` | `driving_track_data`: SAM3 track data for the driving video.<br>`object_indices`: optional comma-separated object indices to keep; blank keeps all tracked objects.<br>`sort_by`: identity ordering strategy: `left_to_right`, `area`, or `none`.<br>`ref_track_data`: optional SAM3 track data for reference identities; do not connect together with `ref_mask`.<br>`ref_mask`: optional plain reference subject `MASK`; do not connect together with `ref_track_data`. | `pose_video_mask` (`IMAGE`), `reference_image_mask` (`IMAGE`) | Renders identity-colored SCAIL-2 RGB semantic masks from SAM3 tracks. Selected objects are mapped deterministically to identity colors in selected order: blue, red, green, magenta, cyan, then yellow. Packed SAM3 masks are unpacked and resized to `orig_size`. The node no longer has a replacement-mode switch; `SCAILPose2SCAIL2Condition.mode` owns mode-specific mask polarity. The preview normally shows colored subjects on black background; it is semantic mask data, not the original video. |

### SCAIL-Pose2 / SCAIL-2

| Node | Main Inputs / Parameters | Outputs | Purpose / Notes |
| --- | --- | --- | --- |
| `SCAILPose2SCAIL2Condition` | Required: `pose_video_mask`, `ref_image`, `ref_mask`, `mode`, `width`, `height`, `num_frames`, and replacement reference-alignment controls.<br>Optional: `pose_video` (historical socket name for the RGB animation driving/conditioning video), `driving_video` (raw RGB replacement video), `additional_ref_image`, `additional_ref_mask`.<br>`mode`: `animation` or `replacement`.<br>`reference_fit_mode`: `auto`, `contain`, `cover`, `fit_height`, or `fit_width`.<br>`reference_anchor`: `auto`, `bottom_center`, or `center`.<br>`reference_target_frame_policy`: `median_bbox`, `first_valid`, or `largest`.<br>`reference_control_region`: `auto`, `subject`, or `upper_subject`.<br>`reference_bbox_margin`, `reference_max_scale`, `reference_min_mask_area_ratio`: replacement-only alignment safety controls. | `condition` (`SCAIL2_CONDITION`) | Builds the validated SCAIL-2 condition payload from RGB video and separate colored semantic masks. Both optional video inputs can stay wired: `animation` uses `pose_video`, while `replacement` requires and uses raw `driving_video`. In `replacement` mode, the node attempts to align `ref_image` and `ref_mask` to `pose_video_mask` before building the condition; in `animation` mode, this reference-geometry alignment is skipped. The condition records identity diagnostics so adapter payloads can report selected driving identity count, reference slot count, and under-provisioned multi-identity references. The declared final canvas, semantic `pose_video_mask`, and `num_frames` must agree; downstream hosts may resize the RGB condition video before VAE encoding. |
| `SCAILPose2PoseMaskGeometryAlign` | `pose_video`: rendered pose image sequence, usually from `RenderNLFPoses.image`.<br>`pose_video_mask`: driving semantic mask sequence from `SCAILPose2ColoredMask.pose_video_mask`. | `pose_video` (`IMAGE`), `summary` (`STRING`) | Scales and translates rendered pose foregrounds so their bbox matches the SAM3-derived driving mask bbox. It infers geometry from the actual image sizes and applies conservative transform-level temporal stabilization for isolated one-frame bbox impulses while preserving continuous motion. Use this for already-rendered pose images or manual animation-mode repair workflows; current `RenderNLFPoses` also has an optional `pose_video_mask` input that applies this alignment inline. |
| `SCAILPose2ReplacementConditionVideo` | `driving_video`: original replacement-mode driving video sequence.<br>`pose_video_mask`: same raw colored semantic mask used by the Condition node.<br>`mask_preset`: `custom`, `tight`, `default`, `loose`, or `soft`; non-custom presets override grow/blur values.<br>`grow_pixels`, `blur_pixels`: custom subject-region expansion and edge softness.<br>`suppression_mode`: `blur_fill`, `mean_fill`, `black_fill`, `white_fill`, or `noise_fill`.<br>`suppression_strength`: `0.0` keeps the original subject; `1.0` fully suppresses subject pixels inside the mask.<br>`noise_seed`: deterministic seed for `noise_fill`. | `driving_video_condition` (`IMAGE`), `summary` (`STRING`) | Legacy/experimental/manual fallback that suppresses original subject pixels before condition encoding. It is not the canonical replacement route because altering this video can weaken the official SCAIL-2 pose-latent motion signal. Prefer raw `driving_video` with the default no-samples route; treat the denoise-mask samples path as a manual background-lock experiment. |
| `SCAILPose2ReplacementDenoiseMask` | `condition`: validated `SCAIL2_CONDITION`.<br>`pose_video_mask`: same raw colored semantic mask used by the Condition node.<br>`mask_preset`: `custom`, `tight`, `default`, `loose`, or `soft`; non-custom presets override grow/blur values.<br>`grow_pixels`: expands the subject denoise area before sampler use.<br>`blur_pixels`: softens the denoise mask edge. | `mask` (`MASK`), `summary` (`STRING`) | Builds a standard ComfyUI `MASK` for replacement/background-lock workflows. The node no longer exposes `strict_replacement_mode` or `invert`; non-replacement modes always emit an all-`1.0` passthrough mask with SCAIL-Pose2 metadata that disables this repo's background-lock samples path in compatible downstream integrations. In `replacement` mode, subject pixels are `1.0` and background preserve pixels are `0.0`; bounded lower-contact refinement is applied internally to improve foot/shoe coverage without adding visible sockets. The summary includes mask coverage, lower-contact, and margin diagnostics. |

## Native SCAIL-2 Workflow Notes

### Data Contract

This README documents this repo's SCAIL-Pose2 nodes only. Downstream wrapper nodes are not listed as repo nodes in `Node Groups`.

Use `SAM3 Video Track.track_data` as the SAM3 source for `SCAILPose2ColoredMask.driving_track_data`. Use `SCAILPose2ColoredMask.pose_video_mask` as `SCAILPose2SCAIL2Condition.pose_video_mask`. Colored Mask `ref_mask` is optional; if no separate reference mask is available, use `SCAILPose2ColoredMask.reference_image_mask` as `SCAILPose2SCAIL2Condition.ref_mask`.

Keep `SCAILPose2SCAIL2Condition.width` and `height` at the final generation size. Do not halve them to match pose latents; downstream integrations may encode pose inputs at the pose-control latent size internally. The adapter handles the required reference-mask and driving-mask packing internally.

SCAIL-2 animation accepts an RGB driving/conditioning video through the historical `pose_video` socket and does not require an NLF or skeleton render. Replacement mode instead selects the raw RGB `driving_video`. In both modes, the colored driving/reference masks remain separate semantic controls; do not substitute the binary replacement denoise mask for them.

`SCAILPose2WanVideoSCAIL2Adapter.condition` is the SCAIL-2 adapter payload intended for compatible downstream SCAIL-2 embedding consumers such as `WanVideoAddSCAIL2ConditionEmbeds`. Context windows and overlap remain downstream generation-wrapper responsibilities, commonly configured through WanVideo Context Options. SCAIL-2 clean-history continuation is not claimed by this repo.

The adapter can report lossy v1 fallback metadata only when degradation is explicitly requested and allowed. Treat that path as compatibility metadata, not full SCAIL-2 parity.

### Multi-Person Identity

For multi-person masks, `SCAILPose2ColoredMask.object_indices` selects which SAM3 objects become SCAIL-2 identities. Blank keeps all objects; `0` keeps the first sorted object; `0,1` keeps the first two sorted objects. `sort_by=left_to_right` is usually the most predictable setup for two-person shots; `area` can change identity order when one person becomes larger.

The selected identity order determines semantic colors, not the raw SAM3 object id: first selected identity is blue, second is red, third is green, then magenta, cyan, and yellow. If you select only one person from a two-person SAM3 track, the output mask should contain only one identity color.

When multiple identities are selected, provide matching reference coverage. A two-person reference can use a multi-subject `ref_image` plus a two-color `ref_mask`, or one base reference plus paired `additional_ref_image` / `additional_ref_mask` slots. The condition payload reports identity diagnostics; under-provisioned reference slots are warnings, not hard errors, because single-reference stylization can still be a deliberate workflow.

### Pose Geometry

SCAIL-2 animation may use any suitable RGB driving/conditioning video. A rendered pose video remains a compatible optional source for a deliberately pose-rendered workflow, but it is not mandatory. When using rendered pose images, they may be half the final generation resolution; keep them as a coordinate-equivalent downsample rather than a different crop or projection.

Legacy SCAIL-v1 pose control still uses rendered pose inputs and may retain `RenderNLFPoses`; the renderer-free statement above applies to SCAIL-2 animation and replacement contracts.

For animation workflows that use this repo's `RenderNLFPoses`, connect its optional `pose_video_mask` input when the rendered skeleton foreground must be scaled and translated to the SAM3-derived mask bbox. That inline path uses the same geometry alignment core as `SCAILPose2PoseMaskGeometryAlign`, including conservative transform-level temporal stabilization for isolated one-frame jumps; the standalone align node is mainly for already-rendered pose images or manual repair.

`RenderNLFPoses` can also consume `NLFPredictPoses.bboxes` through its optional `bboxes` input. Bbox-only repair is used when `pose_video_mask` is not connected. When both are connected, valid mask alignment remains preferred, but empty, frame-mismatched, or bbox-contradicting mask targets are skipped with named log reasons and can fall back to safe bbox repair. In multi-person workflows, connect both `pose_video_mask` and `bboxes`; mask-only routing falls back to NLF/SAM index order when per-person bbox candidates are unavailable. The `render_width` and `render_height` inputs are the source render canvas; after rendering and geometry repair, the node emits `IMAGE` and `MASK` outputs at half of those dimensions.

In multi-person animation workflows, `RenderNLFPoses` uses `pose_video_mask` semantic identity colors to render and align each selected NLF person separately before compositing them. Each identity keeps its own temporal stabilization state, and multi-person bbox ambiguity remains logged as diagnostics instead of being hidden. This avoids the older failure mode where all skeleton foreground was globally aligned into one selected mask bbox. If DWPose face/hand overlay data has an incompatible person count, the overlay is skipped rather than corrupting the repaired body geometry.

For half-resolution pose-control outputs, set `render_width` / `render_height` to the original/source video canvas. The old `width` and `height` inputs were removed from `RenderNLFPoses`; the output pose video is now always `render_width / 2` by `render_height / 2`.

When `dw_poses` is connected, `RenderNLFPoses` renders and repairs the NLF body first, then draws DWPose face/hands at the final half-size output resolution. This prevents drifting 2D face or hand overlays from corrupting the NLF body bbox used for geometry repair.

At runtime, `RenderNLFPoses` logs public-safe provenance and geometry diagnostics such as `module_file=nodes.py`, `nodes_sha256`, the geometry contract version, render source size, mask size, bbox safety, DWPose coordinate bounds, and reference-camera solve safety. The reference `ref_dw_pose` camera solve is kept only when its intrinsics, principal shift, scale, projection residual, and projected coverage are within guardrails; rejected solves fall back to the original camera before rendering.

Replacement mode does not use rendered NLF skeletons as the canonical condition video. Connect the original `driving_video` sequence as raw `driving_video` directly to `SCAILPose2SCAIL2Condition.driving_video` and use `SCAILPose2ColoredMask.pose_video_mask` as the semantic replacement mask. `SCAILPose2PoseMaskGeometryAlign` remains a manual repair utility for already-rendered pose images, not a required replacement-mode step.

### Replacement Mode

#### Required Wiring

Replacement workflows use these repo outputs:

1. `SCAILPose2WanVideoSCAIL2Adapter.condition` for SCAIL-2 conditioning.
2. `SCAILPose2ColoredMask.pose_video_mask` and `reference_image_mask` for semantic subject/reference
   control.

SCAIL-2 conditioning guides subject/reference behavior but does not hard-freeze the original
background. The repository templates deliberately use the no-samples route because it keeps the
generated subject and surrounding scene in one denoising coordinate system. A downstream
`WanVideoEncode` plus `SCAILPose2ReplacementDenoiseMask` samples path remains available for manual
background-lock experiments, but it is not the default template wiring and can create a visible hard
splice when its preserve boundary disagrees with generated geometry. In `animation` mode,
`SCAILPose2ReplacementDenoiseMask` emits an all-`1.0` passthrough mask with metadata that disables
compatible downstream background-lock samples.

For replacement, `SCAILPose2SCAIL2Condition.driving_video` should receive the raw `driving_video` directly. Do not route `RenderNLFPoses` into that input for replacement mode; rendered pose skeletons cannot preserve the original subject proportions relative to the SAM3 mask. Both `pose_video` and `driving_video` can stay wired: the Condition node automatically uses `pose_video` for `animation` mode and `driving_video` for `replacement` mode.

When original subject body shape leaks into replacement output, first verify full SAM3 subject
coverage, raw-driving routing, reference alignment, and SCAIL-2 strength settings. Do not add a
samples/noise-mask path as an automatic leakage fix: it may preserve an incompatible source boundary.
`SCAILPose2ReplacementConditionVideo` is retained only as a legacy/experimental/manual fallback for
unusual experiments, and using it may reduce pose accuracy because it changes the video that becomes
SCAIL-2 pose latents.

Compatible downstream integrations should preserve the replacement mask's SCAIL-Pose2 metadata so subject pixels remain `1.0` replace/denoise areas and background pixels remain `0.0` preserve areas after latent conversion.

For replacement mode, `pose_video_mask` must cover the full driving subject across the driving-video timeline. Partial SAM3 coverage can leave uncovered body, clothing, hair, or limb regions on the preserve side of the downstream samples/noise-mask path, so those regions may continue to inherit the original driving subject even when the reference image and text prompt describe a different person or outfit.

#### Reference And Shape Tuning

The canonical route keeps the replacement condition video raw so the official SCAIL-2 pose-latent
signal remains intact and leaves sampler `samples` disconnected. If you deliberately use the legacy
`SCAILPose2ReplacementConditionVideo` fallback, treat it as an experiment after confirming the raw
condition and semantic masks are correct; a conservative starting point is `mask_preset=custom`,
`grow_pixels=8`, `blur_pixels=0`, `suppression_mode=blur_fill`, and
`suppression_strength=1.0`.

When the reference image has a different crop, aspect ratio, or subject scale from the driving subject, keep `ref_image` and `ref_mask` connected directly to `SCAILPose2SCAIL2Condition`. In `replacement` mode, Condition automatically attempts to align the reference image/mask to `pose_video_mask` before building the condition. In `animation` mode, this replacement-only alignment path is skipped.

Reference alignment controls are replacement-only:

- `reference_control_region=auto` chooses whole-subject alignment for tall full-body masks and upper-subject local alignment for portrait or half-body masks. Use `subject` to force the previous whole-bbox route, or `upper_subject` to force the local portrait/upper-body route.
- `reference_fit_mode=auto` maps to `contain` for whole-subject alignment and `cover` for upper-subject alignment. Explicit `contain`, `cover`, `fit_height`, and `fit_width` override that policy.
- `reference_anchor=auto` maps to `bottom_center` for whole-subject alignment and `center` for upper-subject alignment. Explicit `bottom_center` and `center` override that policy.
- `reference_target_frame_policy=median_bbox` uses the median driving bbox across valid frames; `first_valid` and `largest` are available when a workflow needs a specific target-frame strategy.
- `reference_bbox_margin`, `reference_max_scale`, and `reference_min_mask_area_ratio` limit unsafe placement, oversized scaling, or nearly-empty masks. If alignment validation fails, Condition logs a warning and falls back to the direct `ref_image` / `ref_mask` pair.

Compatible WanVideoWrapper builds may expose SCAIL-2 strength controls on their SCAIL-2 condition-embeds node: `ref_image_strength`, `ref_mask_strength`, `condition_video_strength`, and `driving_mask_strength`. Their default `1.0` values preserve previous behavior. To bias replacement toward the reference image, raise `ref_image_strength` carefully and reduce `condition_video_strength` only when original body shape still dominates.

#### Preview Behavior

Early sampler previews can still show a noisy or incomplete scene. Judge composition and boundaries
from later previews or the final decoded output, not from the first preview frames alone. If an
optional background-lock samples experiment is enabled, compare it against the default no-samples
template and inspect the entire subject/background intersection for hard-splice displacement.

### Troubleshooting

**Empty SAM3 tracking masks:** inspect the tracking result and object selection before tuning
Colored Mask. Empty tracks intentionally produce background masks, and a missing reference mask
uses the documented solid reference fallback. SAM3 detection belongs to ComfyUI core; check the
active core's detector handling of segmentation features after feature-pyramid scalping when
comparing host builds. Updating this node pack alone does not update the core detector. Verify
the active installation and rerun tracking before judging the downstream mask preview.

**Action inaccuracy:** for replacement mode, keep `SCAILPose2SCAIL2Condition.driving_video` wired to raw `driving_video`. Do not suppress, repaint, or replace that video before SCAIL-2 condition encoding; doing so can weaken the official pose-latent motion signal.

**Source leakage:** verify SAM3 coverage, raw `driving_video` routing, reference alignment, prompt,
and SCAIL-2 strengths first. The default templates intentionally leave sampler `samples`
disconnected. Only test encoded driving samples as an explicit A/B experiment, because a hard
preserve boundary can clip newly generated geometry.

**Stale runtime wrapper copy:** if behavior does not match this README after updating files, confirm
the active ComfyUI custom-node folder is using a compatible ComfyUI-WanVideoWrapper build with
native SCAIL-2 condition embeds. An older copied or cached wrapper can miss that payload contract or
ignore SCAIL-Pose2 mask metadata used by the optional background-lock lane.

**Optional background-lock diagnostics:** compatible wrapper builds may log
`noise_mask_latent_contract`, `samples_initialization_contract`, and
`samples_window_alignment_contract` when the manual samples lane is enabled. Check `subject_ratio`,
`preserve_ratio`, `latent_grow_pixels`, `latent_temporal_grow_frames`, `subject_source`, and
`preserve_source`, then compare the final frame against the default no-samples template. These logs
do not make the optional hard-splice route canonical.

**Prompt and clothing mismatch:** Prompt clothing detail cannot compensate for an undercovered driving-subject mask. If replacement output changes clothing randomly or returns to the driving-video person mid-generation, first fix SAM3 object selection and `pose_video_mask` coverage so the full driving subject is in the denoise/replace region before tuning prompt wording or SCAIL-2 strength controls.

**Render NLF geometry drift:** first confirm the active `RenderNLFPoses` log includes the expected `nodes_sha256` and geometry contract fields after updating the node pack. Then keep `render_width` and `render_height` on the source video canvas, temporarily disconnect `ref_dw_pose`, disable DWPose face/hand overlays, test bbox-only repair by disconnecting `pose_video_mask`, and reconnect `pose_video_mask` only after its mask size and bbox diagnostics match the source canvas.

**Reference local mismatch:** if replacement improves global scale but the head, face, or upper-body region is still offset, test `reference_control_region=upper_subject`. For full-body references where feet or whole-body height matters more than face placement, test `reference_control_region=subject`, `reference_fit_mode=contain`, and `reference_anchor=bottom_center`. Current reference alignment is mask-geometry based; it does not run face-landmark registration.

**Multi-person identity mismatch:** if only one person appears in the colored mask, verify SAM3 prompt/object settings first, then check `object_indices`. If two people are selected but reference identity transfer is weak, use a two-color `ref_mask` or add paired `additional_ref_image` / `additional_ref_mask` entries so the condition has enough reference slots.

### Notes

- Colored Mask previews show colored subjects on black background because they are semantic masks, not original video frames.
- Reference identity depends on `ref_image`, `ref_mask` / `reference_image_mask`, and the downstream reference embedding path.
- Long-running nodes log safe progress/log summaries such as shape, dtype, frame count, object count, and elapsed time.
- The Condition node does not expose SCAIL-2 segment or continuation controls; downstream context windows are owned by the generation wrapper.
- This repo's restored NLF pose loader returns `NLF_MODEL`; the WanVideoWrapper NLF family may expose `NLFMODEL` through nodes such as `(Download)Load NLF Model`. Those model contracts are not directly wire-compatible: this repo expects local NLF `.safetensors` assets, while wrapper-side NLF loaders may manage `.torchscript` assets and downloads.

## License

MIT License
