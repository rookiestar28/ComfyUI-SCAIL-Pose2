"""ComfyUI node wrappers for WanVideo adapter payloads."""

from __future__ import annotations

from .scail2.wanvideo_adapter import build_wan_scail_images_payload


class SCAILPose2WanSCAILImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "pose_images": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 1, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "step": 1}),
                "num_frames": ("INT", {"default": 81, "min": 1, "step": 1}),
            },
            "optional": {
                "clip_ref_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = (
        "SCAIL_WAN_SCAIL_IMAGES",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "adapter_payload",
        "ref_image",
        "pose_images",
        "clip_ref_image",
        "width",
        "height",
        "num_frames",
    )
    FUNCTION = "build"
    CATEGORY = "SCAIL-Pose2/WanVideoWrapper"
    DESCRIPTION = "Validate and pass through images for current WanVideoWrapper SCAIL nodes."

    def build(
        self,
        ref_image,
        pose_images,
        width,
        height,
        num_frames,
        clip_ref_image=None,
    ):
        payload = build_wan_scail_images_payload(
            ref_image=ref_image,
            pose_images=pose_images,
            width=width,
            height=height,
            num_frames=num_frames,
            clip_ref_image=clip_ref_image,
        )
        return (
            payload,
            payload["ref_image"],
            payload["pose_images"],
            payload["clip_ref_image"],
            payload["width"],
            payload["height"],
            payload["num_frames"],
        )


NODE_CLASS_MAPPINGS = {
    "SCAILPose2WanSCAILImages": SCAILPose2WanSCAILImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCAILPose2WanSCAILImages": "SCAIL-Pose2 Wan SCAIL Images",
}
