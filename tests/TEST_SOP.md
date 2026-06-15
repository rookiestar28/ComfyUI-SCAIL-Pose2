# Test SOP

This document is the source-of-truth local verification workflow for **ComfyUI-SCAIL-Pose2**.

## Repository Facts

- This repository is a Python ComfyUI custom-node project in preparation/reconstruction.
- The target product is `comfyui-scail-pose2`, a preprocessing and adapter node pack for SCAIL/SCAIL-2 workflows.
- The intended downstream generation owner is `ComfyUI-WanVideoWrapper`; this repository should not duplicate WanVideoWrapper sampler/model-loader/decode responsibilities unless a later roadmap item explicitly approves that direction.
- The tracked package currently includes restored v1 pose nodes, SCAIL-2 helper modules, WanVideoWrapper adapter nodes, workflow skeletons, packaging metadata, test runners, and release validation tests.
- Ignored reference material is read-only and must not be executed during routine validation.
- There is no tracked `package.json`, frontend extension, Playwright config, or browser test harness in the current repo state.
- The active internal roadmap defines staged implementation items.

## Required Reading Order

Before running validation for non-documentation work, read:

1. `tests/TEST_SOP.md`
2. `tests/E2E_TESTING_NOTICE.md`
3. `tests/E2E_TESTING_SOP.md`

Also read the internal implementation-plan SOP before creating or closing any non-documentation implementation plan or record.

## Acceptance Rule

A non-documentation change is not accepted until required checks pass and evidence is recorded in the implementation record and repo-local command log.

Required gate for this repository after product code exists:

1. Secret scan: `pre-commit run detect-secrets --all-files`
2. Pre-commit hooks: `pre-commit run --all-files --show-diff-on-failure`
3. Python compile/import smoke checks for tracked product modules
4. Focused unit or tensor/contract behavior checks for changed nodes
5. ComfyUI custom-node smoke/integration lane per `tests/E2E_TESTING_SOP.md`

If `.pre-commit-config.yaml`, product modules, or test harness files are missing for a code-changing item, record the blocker and complete the roadmap/test-infrastructure item before claiming `ACCEPTED`, unless the user explicitly approves a fallback.

## Problem-First Test Design Rule

All tests and validation flows must be designed to catch real defects, regressions, drift, and broken assumptions before users hit them.

For every bugfix or high-risk change, start from:

```text
Which test would have caught this before release?
```

Acceptance evidence for bugfix/hotfix work must follow `Reproduce -> Pin -> Sweep`:

- Reproduce: capture pre-fix failure evidence.
- Pin: add or update targeted regression coverage for the root cause.
- Sweep: run the required broader gate after the targeted fix passes.

A green full gate alone is not sufficient bugfix evidence.

## Documentation-only Exception

If all touched files are documentation/planning text only and no code, tests, scripts, dependency manifests, runtime config, generated artifacts, or public release artifacts changed, full runtime test execution is optional.

Required evidence for documentation-only changes:

1. list touched documentation files
2. confirm no product code, tests, scripts, packaging, or runtime config were changed
3. run a lightweight document check where practical, such as:

```powershell
git diff --check -- <changed-doc-paths>
```

This exception does not apply once any `.py`, test, script, packaging, dependency, or runtime configuration file changes.

## Environment Policy

Use one Python interpreter consistently across all stages.

Recommended interpreter order:

1. active ComfyUI Python environment
2. Windows repo-local `.venv`
3. WSL/Linux repo-local `.venv-wsl`

For future product-code validation, Python 3.10+ is required unless ComfyUI or dependency compatibility requires a narrower version.

Do not rely on global Python packages as the first-choice validation path for accepted implementation work.

## Current Product Module Discovery

Do not hard-code stale module names from another repository.

When product code exists, derive the compile/import target set from tracked source files such as:

- root `__init__.py`
- root `nodes.py`
- `nodes_*.py`
- package directories such as `scail2/`
- any future modules explicitly exported by `NODE_CLASS_MAPPINGS`

Do not compile or execute ignored reference material as part of the product module set.

Example PowerShell discovery command for review only:

```powershell
@(
  git ls-files "*.py"
  git ls-files --others --exclude-standard "*.py"
) | Where-Object { $_ -and $_ -notmatch '^(reference|\\.planning|\\.sessions)/' } | Sort-Object -Unique
```

## One-command Full Test Scripts

The repository provides repo-local full test runners. Prefer them for acceptance validation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_tests_windows.ps1
```

```bash
bash scripts/run_full_tests_linux.sh
```

These scripts use repo-local virtual environments, install `pre-commit` there when missing, run the required secret/pre-commit/unit-test gate, and report frontend E2E as not applicable unless a tracked frontend harness is added.

If these scripts are absent, use the manual staged workflow below and record that no one-command runner exists.

## Manual Staged Workflow

### 1. Workspace and applicability check

```powershell
git status --short
@(
  git ls-files "*.py"
  git ls-files --others --exclude-standard "*.py"
) | Sort-Object -Unique
Test-Path .pre-commit-config.yaml
Test-Path package.json
```

Record whether the change is documentation-only, Python product code, tests/scripts, packaging, or runtime configuration.

### 2. Secret scan

```powershell
pre-commit run detect-secrets --all-files
```

If pre-commit infrastructure is missing, record the blocker. Code-changing work cannot be marked accepted until the infrastructure is restored or a user-approved fallback is documented.

If the `pre-commit` console command is not on `PATH` but the Python package is installed, use this equivalent form and record the reason:

```powershell
python -m pre_commit run detect-secrets --all-files
```

### 3. Pre-commit hooks

```powershell
pre-commit run --all-files --show-diff-on-failure
```

Equivalent fallback when the console command is unavailable:

```powershell
python -m pre_commit run --all-files --show-diff-on-failure
```

If hooks modify files, review the changes and rerun until clean.

### 4. Python compile/import smoke

Use the same interpreter for all Python checks.

When product modules exist:

```powershell
$files = @(
  git ls-files "*.py"
  git ls-files --others --exclude-standard "*.py"
) | Where-Object { $_ -and $_ -notmatch '^(reference|\\.planning|\\.sessions)/' } | Sort-Object -Unique
if ($files.Count -gt 0) { python -m py_compile @files }
```

Import smoke should verify package-level node registration when a root package exists:

```powershell
python -c "import importlib.util, pathlib; p=pathlib.Path('__init__.py'); assert p.exists(), 'missing __init__.py'; spec=importlib.util.spec_from_file_location('ComfyUI_SCAIL_Pose2', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); assert hasattr(m, 'NODE_CLASS_MAPPINGS'); print(sorted(m.NODE_CLASS_MAPPINGS.keys()))"
```

If ComfyUI runtime modules such as `folder_paths` are required and unavailable, record whether the node import should be lazy/mockable or whether a ComfyUI environment is required.

### 5. Unit and contract tests

Run tracked tests when present:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Future tests should cover:

- existing v1 node key/display compatibility
- OpenPose/DWPose conversion behavior
- NLF render shape and metadata contracts with synthetic inputs where possible
- SCAIL-2 RGB mask palette validation
- 28-channel mask latent packing
- `SCAIL2_CONDITION` validation
- WanVideoWrapper contract mapping with mocks, not runtime imports
- SAM3 missing-dependency behavior through lazy imports

### 6. ComfyUI custom-node smoke/integration lane

Follow `tests/E2E_TESTING_SOP.md`.

## Frontend / npm / Playwright Policy

`npm test`, Playwright, and browser E2E are not applicable unless a tracked frontend harness and `package.json` are added.

If a future frontend harness is added:

- update this SOP and `tests/E2E_TESTING_SOP.md`
- verify Node.js 18+
- document the browser E2E command
- keep the ComfyUI custom-node smoke lane for node behavior

## Evidence Recording

Implementation records and command logs must include:

- date and timezone
- OS and shell
- branch and commit SHA
- Python executable and version
- Node/npm versions when frontend/E2E applies
- relevant package versions such as `torch`, `numpy`, and `Pillow` when used
- exact commands
- pass/fail/blocked status for every required stage
- reason for any repo-specific override or non-applicability decision

## Failure Handling

- If a check fails, fix the root cause and rerun the failed check plus dependent checks.
- If a check is blocked by missing infrastructure, record the blocker and reference the roadmap item that owns it.
- Do not mark code-changing work accepted while required gate infrastructure is blocked.
- Do not execute ignored reference material to make validation pass.
