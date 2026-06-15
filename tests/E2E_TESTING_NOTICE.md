# E2E Testing Notice

All integration or end-to-end validation for this repository must follow `tests/E2E_TESTING_SOP.md`.

## Repo-specific Scope

This repository is a ComfyUI custom-node project, not a standalone web application.

The current repo state does not contain:

- frontend JavaScript or TypeScript extension files
- `package.json`
- Playwright configuration
- a browser-based test harness

Therefore `npm test` and Playwright browser E2E are not applicable for the current repo state.

## Required Replacement Lane

Use the ComfyUI custom-node smoke/integration lane in `tests/E2E_TESTING_SOP.md`.

That lane verifies the node pack at the user-facing ComfyUI integration boundary:

- package import behavior
- node registration
- changed-node runtime behavior with deterministic inputs
- image/tensor/mask/condition payload contracts
- optional dependency failure behavior
- WanVideoWrapper adapter contract compatibility through mocks/static workflow checks

## Mandatory Test Design Rule

E2E and integration checks must be designed to reproduce real user-visible failures and catch bugs early, not merely to pass validation.

For every user-reported or high-risk regression, ask which integration assertion would have caught it before release, then add or update that assertion.

Route-load-only or import-only evidence is not sufficient for changed node behavior. Include at least one assertion against the final output contract of the changed node or adapter.

## Exception

Strict documentation-only changes do not require entering the E2E workflow.

Once product code, tests, scripts, packaging, dependency manifests, generated artifacts, or runtime configuration changes, this exception does not apply.

## Evidence Requirement

Implementation records must state one of:

- `E2E lane passed`
- `E2E lane not applicable: documentation-only change`
- `E2E lane blocked`, with the exact missing dependency or infrastructure

If a future frontend harness is added, this notice must be updated to identify both:

- the browser/frontend E2E lane
- the ComfyUI custom-node smoke/integration lane
