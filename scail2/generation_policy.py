"""Generation ownership policy for SCAIL-Pose2."""

from __future__ import annotations


DECISION_DEFER_TO_WANVIDEOWRAPPER = "DEFER_TO_WANVIDEOWRAPPER"
DECISION_IMPLEMENT_EXPERIMENTAL_DIRECT_GENERATOR = (
    "IMPLEMENT_EXPERIMENTAL_DIRECT_GENERATOR"
)
DECISION_BLOCKED_PENDING_WRAPPER_SUPPORT = "BLOCKED_PENDING_WRAPPER_SUPPORT"

ALLOWED_DIRECT_GENERATION_DECISIONS: tuple[str, ...] = (
    DECISION_DEFER_TO_WANVIDEOWRAPPER,
    DECISION_IMPLEMENT_EXPERIMENTAL_DIRECT_GENERATOR,
    DECISION_BLOCKED_PENDING_WRAPPER_SUPPORT,
)

DIRECT_GENERATION_DECISION = DECISION_DEFER_TO_WANVIDEOWRAPPER

SCAIL2_DIRECT_GENERATION_REQUIREMENTS: tuple[str, ...] = (
    "reference_mask",
    "driving_mask",
    "mask_latents_28_channel",
    "replacement_flag",
    "additional_reference_masks",
    "long_video_clean_history",
    "lora_support",
    "segment_overlap",
)

DIRECT_GENERATION_NODE_EXCLUSIONS: tuple[str, ...] = (
    "SCAIL2ModelLoader",
    "SCAIL2Generate",
    "SCAIL2PipelineGenerate",
)

REQUIRED_DIRECT_GENERATOR_PLAN_GATES: tuple[str, ...] = (
    "lazy_optional_imports",
    "model_path_validation",
    "mocked_pipeline_tests",
    "optional_real_smoke_conditions",
    "rollback_plan",
    "full_gate_evidence",
)


def direct_generation_enabled() -> bool:
    return DIRECT_GENERATION_DECISION == DECISION_IMPLEMENT_EXPERIMENTAL_DIRECT_GENERATOR
