# ComfyUI-SCAIL-Pose2

ComfyUI-SCAIL-Pose2 is a ComfyUI custom node package for SCAIL and SCAIL-2 pose and mask preprocessing. It prepares pose images, DWPose-compatible pose data, NLF pose renders, SCAIL-2 condition metadata, and adapter payloads for wrapper-side generation nodes. It is designed to work with ComfyUI-WanVideoWrapper as the downstream generation owner.

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

- `SCAILPose2WanVideoSCAIL2Adapter`: converts a validated `SCAIL2_CONDITION` into a versioned SCAIL-2 adapter payload (`SCAIL2_WANVIDEO_PAYLOAD`). The payload preserves SCAIL-2 semantic mask metadata and explicitly marks current WanVideoWrapper gaps; it is not live wrapper-side full SCAIL-2 parity. When `degrade_to_v1` and `allow_degradation` are both enabled, the node also exposes current WanVideoWrapper v1-compatible reference image, pose image, and size/frame outputs for users who accept the documented semantic losses.

### SCAIL-Pose2 / SAM3

- `SCAIL2SAM3DependencyCheck`: checks whether optional SAM3 preprocessing dependencies are available in the active ComfyUI environment without making SAM3 a startup requirement.
- `SCAILPose2ColoredMask`: renders identity-colored SCAIL-2 RGB masks from existing SAM3 track data. It supports shared identity sorting, object filtering, animation/replacement background semantics, optional reference track data, and optional plain reference masks.

### SCAIL-Pose2 / SCAIL-2

- `SCAILPose2SCAIL2Condition`: builds a validated `SCAIL2_CONDITION` payload from reference images, pose video, RGB semantic reference and driving masks, mode, segment settings, optional additional reference pairs, and continuation metadata. Supported modes are `animation`, `replacement`, and `pose_driven`.

## License

MIT License
