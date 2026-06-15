# ComfyUI-SCAIL-Pose2

ComfyUI-SCAIL-Pose2 is a ComfyUI custom node package for SCAIL and SCAIL-2 pose and mask preprocessing. It prepares pose images, DWPose-compatible pose data, NLF pose renders, SCAIL-2 condition metadata, and adapter payloads for wrapper-side generation nodes. It is designed to work with ComfyUI-WanVideoWrapper as the downstream generation owner.

## Current Scope

- v1 SCAIL pose-control nodes: load NLF models, predict NLF poses, convert OpenPose or ViTPose keypoints to DWPose-style data, render NLF poses, and export NLF poses as 3D animation.
- WanVideoWrapper adapter node: validate and pass reference, pose, optional clip reference, size, and frame-count payloads for the current wrapper SCAIL image path.
- SCAIL-2 condition and mask preparation: build typed condition data for RGB semantic masks, reference and driving mask indices, replacement mode, segment settings, and unsupported-feature reporting.
- SAM3 preprocessing boundary: SAM3 support is optional and lazy; base package import does not require SAM3 dependencies.
- WanAnimate fallback helper: convert SCAIL-2 condition data into an explicitly lossy WanAnimate-compatible fallback only when semantic-mask degradation is accepted.

Direct generation is deferred. Model loading, sampling, VAE work, decoding, memory controls, and video output remain owned by ComfyUI-WanVideoWrapper.

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

## Model Folders

Place NLF pose model files in:

```text
ComfyUI/models/nlf/
```

Detection model files used by compatible pose preprocessors belong in:

```text
ComfyUI/models/detection/
```

Model checkpoints are not bundled with this package. Do not hard-code local absolute model paths in workflows; use ComfyUI model folders so workflows remain portable.

## Optional Dependencies

ComfyUI provides the core runtime, including Torch in normal installations. This package keeps heavy or workflow-specific components lazy so ComfyUI can start even when optional stacks are absent.

- Taichi is optional. If Taichi rendering is selected but unavailable, NLF rendering falls back to the Torch renderer.
- SAM3 preprocessing is optional. SAM3-specific imports are attempted only when the SAM3 execution path runs; missing dependency errors should point to the optional SAM3 requirement instead of breaking package import.
- ComfyUI-WanVideoWrapper is the intended downstream generation partner, but this package does not import it at startup.

## Node Groups

### SCAIL-Pose

- `NLFModelLoader`
- `NLFPredictPoses`
- `ConvertOpenPoseKeypointsToDWPose`

### WanAnimatePreprocess

- `PoseDetectionVitPoseToDWPose`

### WanVideoWrapper

- `RenderNLFPoses`
- `SaveNLFPosesAs3D`

### SCAIL-Pose2 / WanVideoWrapper

- `SCAILPose2WanSCAILImages`

### SCAIL-Pose2 / SAM3

- `SCAIL2SAM3DependencyCheck`

## WanVideoWrapper Pipeline Boundary

Use ComfyUI-SCAIL-Pose2 to prepare pose and condition inputs. Use ComfyUI-WanVideoWrapper for Wan model loading, sampling, decoding, and final video output.

Current WanVideoWrapper SCAIL image compatibility is v1-style reference and pose image conditioning. Full SCAIL-2 RGB semantic mask consumption is represented in condition data and static workflow skeletons, but it is not claimed as direct wrapper parity until wrapper-side support exists.

## WanAnimate Fallback

The WanAnimate fallback path is a controlled degradation path. It can collapse SCAIL-2 semantic mask information into a simpler mask shape for wrapper paths that cannot consume full SCAIL-2 RGB semantic data.

This fallback is not full SCAIL-2 parity. Use it only when the loss of semantic channel detail is acceptable for the workflow.

## Development Validation

For repository validation, use the included full test runner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_tests_windows.ps1
```

Linux and WSL users can run:

```bash
bash scripts/run_full_tests_linux.sh
```

Frontend npm or Playwright tests are not part of the current package because this repository does not ship a tracked frontend harness.

## License

MIT License
