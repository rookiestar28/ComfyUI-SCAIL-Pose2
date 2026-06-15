# E2E Testing SOP

This SOP defines the ComfyUI custom-node smoke/integration workflow for **ComfyUI-SCAIL-Pose2**.

## Scope

This is not currently a frontend Playwright workflow.

The E2E boundary for this repository is the ComfyUI custom-node integration boundary:

- ComfyUI or a compatible test harness can discover and import the node package.
- `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` expose expected nodes.
- Changed node classes execute representative workflows with deterministic inputs.
- Output `IMAGE`, `MASK`, `NLFPRED`, `SCAIL2_CONDITION`, and adapter metadata contracts match node definitions.
- Optional dependencies fail lazily with actionable messages.
- WanVideoWrapper compatibility is validated by static contracts and mocks unless the user explicitly approves live wrapper execution.

## Problem-First Test Design Rule

E2E scripts and mocked harness flows must be designed to reproduce failures and catch bugs early.

Prefer assertions that prove final user-visible behavior, payload shape, state synchronization, contract compatibility, and failure feedback. Avoid pass-only checks that only prove the package imported or a mocked happy path returned.

## Requirements

Required for code-changing validation:

- Python 3.10+ in the same environment used by ComfyUI, or a repo-local venv with equivalent dependencies
- product package files restored or implemented
- test dependencies required by the changed surface

Required for image/tensor node checks:

- `torch`
- `numpy`
- `Pillow`

Potentially required when the changed surface uses them:

- ComfyUI runtime modules such as `folder_paths` and `comfy.model_management`
- NLF model dependencies
- optional SAM3 dependencies
- WanVideoWrapper only for an explicitly approved live integration check

Not required in the current repo state:

- Node.js
- npm
- Playwright

## General Procedure

1. Confirm the changed surface:
   - documentation-only
   - node registration/import
   - pose preprocessing
   - SCAIL-2 mask/condition utilities
   - SAM3 optional preprocessing
   - WanVideoWrapper adapter contracts
   - direct generation fallback, if ever approved
2. Use the same Python interpreter throughout the lane.
3. Prefer small deterministic tensors/images over large media.
4. Do not execute ignored reference material.
5. Do not require private media, private prompts, local absolute model paths, or gated model credentials.

## Windows Procedure

Use the active ComfyUI Python or repo-local `.venv`.

```powershell
python --version
@'
import sys
print(sys.executable)
'@ | python -
```

Check optional tensor dependencies when needed:

```powershell
@'
for name in ["torch", "numpy", "PIL"]:
    try:
        mod = __import__(name)
        print(name, getattr(mod, "__version__", "importable"))
    except Exception as exc:
        print(f"{name} unavailable: {exc}")
'@ | python -
```

Compile tracked product Python files when they exist:

```powershell
$files = @(
  git ls-files "*.py"
  git ls-files --others --exclude-standard "*.py"
) | Where-Object { $_ -and $_ -notmatch '^(reference|\\.planning|\\.sessions)/' } | Sort-Object -Unique
if ($files.Count -gt 0) { python -m py_compile @files }
```

Run tracked tests when present:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Linux / WSL Procedure

Use the active ComfyUI Python or repo-local `.venv-wsl`.

```bash
python --version
python - <<'PY'
import sys
print(sys.executable)
PY
```

Compile tracked product Python files when they exist:

```bash
python - <<'PY'
import subprocess, sys
files = []
for cmd in (
    ["git", "ls-files", "*.py"],
    ["git", "ls-files", "--others", "--exclude-standard", "*.py"],
):
    files.extend(subprocess.check_output(cmd, text=True).splitlines())
files = sorted(set(files))
files = [p for p in files if not p.startswith(("reference/", ".planning/", ".sessions/"))]
if files:
    subprocess.check_call([sys.executable, "-m", "py_compile", *files])
else:
    print("no tracked product python files")
PY
```

Run tracked tests when present:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Node Registration Smoke

When the root package exists, verify node registration.

Local file import smoke:

```powershell
python -c "import importlib.util, pathlib; p=pathlib.Path('__init__.py'); assert p.exists(), 'missing __init__.py'; spec=importlib.util.spec_from_file_location('ComfyUI_SCAIL_Pose2', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); assert hasattr(m, 'NODE_CLASS_MAPPINGS'); print(sorted(m.NODE_CLASS_MAPPINGS.keys()))"
```

Expected future v1 compatibility keys:

```text
NLFModelLoader
NLFPredictPoses
PoseDetectionVitPoseToDWPose
ConvertOpenPoseKeypointsToDWPose
RenderNLFPoses
SaveNLFPosesAs3D
```

If full import fails because ComfyUI modules are unavailable, use one of these approved approaches:

- run from the ComfyUI Python environment
- add a focused test stub for ComfyUI-only modules
- record the import-context blocker and run changed-node contract tests that do not need full ComfyUI runtime

Do not import or execute WanVideoWrapper just to prove SCAIL-Pose2 base registration.

## Changed-node Assertion Requirements

Each implementation must include assertions matching the changed node type:

- v1 pose nodes: node keys, input/output type declarations, conversion output shape, render output shape, and metadata stability
- `IMAGE` outputs: tensor shape, channel count, dtype/value range, and deterministic placement
- `MASK` outputs: mask shape, value range, and frame alignment
- `SCAIL2_CONDITION`: mode, frame count, width/height, mask palette, source kind, wrapper target metadata, and unsupported-feature reporting
- SCAIL-2 masks: seven RGB colors, black/background handling, threshold behavior, and 28-channel temporal packing
- SAM3 nodes: mocked success path, missing dependency path, missing model path, and track-count mismatch behavior
- WanVideoWrapper adapters: mocked socket names/types, unsupported SCAIL-2 feature reporting, and no runtime wrapper import
- file output nodes: isolated path behavior and no writes outside the workspace or ComfyUI output directory

## WanVideoWrapper Contract Validation

Use static/mocked validation by default.

Required assertions for adapter work:

- adapter output names and ComfyUI types match the planned wrapper target sockets
- current wrapper SCAIL path is treated as v1-style reference/pose conditioning only
- SCAIL-2-only fields are either preserved in `SCAIL2_CONDITION` or explicitly listed as unsupported by the current wrapper path
- no adapter imports WanVideoWrapper at package import time

Live WanVideoWrapper execution is out of normal scope. It requires explicit approval, a disposable environment, no secrets, and no execution of untrusted reference setup scripts.

## Non-applicable Frontend E2E

Do not run these commands for the current repo state:

```bash
npm install
npx playwright install chromium
npm test
```

They become applicable only after a tracked frontend harness and `package.json` are added.

## Troubleshooting

- `ModuleNotFoundError: folder_paths`: run from the ComfyUI environment, provide a test stub, or record that full ComfyUI import smoke is blocked by missing runtime context.
- `ModuleNotFoundError: torch`: use the ComfyUI Python environment or install dependencies into the repo-local venv.
- Optional dependency import failure: verify that the dependency is imported lazily and only required when the relevant node executes.
- Large media memory pressure: replace large media with tiny deterministic tensors for automated checks.
- Reference repo temptation: do not execute ignored reference scripts; borrow behavior only through read-only analysis and mocked contracts.

## Evidence Recording

Record:

- exact Python executable
- Python version
- relevant package versions
- command log path
- commands run
- assertion coverage summary
- final pass/fail/blocked status
- any non-applicable frontend/E2E rationale
