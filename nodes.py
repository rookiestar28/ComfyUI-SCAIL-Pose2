import copy
import datetime
import hashlib
import logging
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
_RENDER_NLF_GEOMETRY_CONTRACT_VERSION = "render_nlf_pose_geometry_v3"
_RENDER_NLF_RUNTIME_PROVENANCE_CACHE = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

try:
    import folder_paths
    _FOLDER_PATHS_AVAILABLE = True
except ImportError:
    _FOLDER_PATHS_AVAILABLE = False

    class _FolderPathsFallback:
        models_dir = os.path.join(script_directory, "models")

        @staticmethod
        def add_model_folder_path(_name, _path):
            return None

        @staticmethod
        def get_filename_list(_name):
            return []

        @staticmethod
        def get_full_path_or_raise(name, value):
            raise ImportError(
                f"ComfyUI folder_paths is required to resolve {name!r}: {value!r}"
            )

        @staticmethod
        def get_output_directory():
            return os.path.join(script_directory, "output")

    folder_paths = _FolderPathsFallback()

try:
    import comfy
    from comfy import model_management as mm
    import comfy.ops
    import comfy.model_patcher
except ImportError:
    comfy = None
    mm = None

try:
    from comfy.utils import ProgressBar
except ImportError:
    class ProgressBar:
        def __init__(self, _total):
            self.total = _total

        def update(self, _value):
            return None

        def update_absolute(self, _value):
            return None

from .scail2.nlf_geometry import (
    align_pose_video_to_bboxes,
    bbox_payload_is_safe_for_render_repair,
    format_nlf_render_bbox_diagnostics,
    format_nlf_source_canvas_diagnostics,
    normalize_nlf_bboxes,
    pose_mask_alignment_is_safe_for_render_repair,
    resize_bhwc_video,
    resize_mask_video,
    select_nlf_bboxes_for_identity,
    validate_ref_dwpose_camera_solve,
)
from .scail2.geometry import frame_bboxes
from .scail2.identity import (
    identity_count_from_semantic_mask,
    semantic_identity_rgb_mask,
)
from .scail2.pose_alignment import align_pose_video_to_mask


def _require_dependency(name, module):
    if module is None:
        raise ImportError(
            f"{name} is required for this node execution. "
            "Install the dependency in the active ComfyUI environment."
        )
    return module


def _render_nlf_runtime_provenance():
    global _RENDER_NLF_RUNTIME_PROVENANCE_CACHE
    if _RENDER_NLF_RUNTIME_PROVENANCE_CACHE is not None:
        return _RENDER_NLF_RUNTIME_PROVENANCE_CACHE
    try:
        with open(__file__, "rb") as handle:
            nodes_hash = hashlib.sha256(handle.read()).hexdigest()[:12]
    except OSError:
        nodes_hash = "unavailable"
    _RENDER_NLF_RUNTIME_PROVENANCE_CACHE = (
        "scail_pose2_runtime "
        "module_file=nodes.py "
        f"nodes_sha256={nodes_hash} "
        f"geometry_contract={_RENDER_NLF_GEOMETRY_CONTRACT_VERSION} "
        "half_output=True "
        "source_canvas_diagnostics=True "
        "mask_bbox_arbitration=True "
        "ref_camera_guardrails=True"
    )
    return _RENDER_NLF_RUNTIME_PROVENANCE_CACHE


def _overlay_dwpose_2d_on_frames(
    *,
    frames_tensor,
    mask,
    dw_pose_input,
    draw_face,
    draw_hands,
    identity_count=None,
    alignment_transforms_by_person=None,
    alignment_source_width=None,
    alignment_source_height=None,
):
    if dw_pose_input is None or (not draw_face and not draw_hands):
        return frames_tensor, mask
    _require_dependency("numpy", np)
    _require_dependency("torch", torch)

    from .pose_draw.draw_pose_utils import draw_pose_to_canvas_np

    output_height = int(frames_tensor.shape[1])
    output_width = int(frames_tensor.shape[2])
    overlay_dw_pose_input = _transform_dwpose_sequence_coordinates(
        dw_pose_input,
        transforms_by_person=alignment_transforms_by_person,
        source_width=alignment_source_width,
        source_height=alignment_source_height,
    )
    person_count = _dwpose_person_count(dw_pose_input)
    if identity_count is not None and int(identity_count) > 0:
        expected = int(identity_count)
        if person_count not in {0, expected}:
            logging.warning(
                "Render NLF Poses skipped DWPose overlay: identity/person "
                "count mismatch identities=%s dwpose_persons=%s",
                expected,
                person_count,
            )
            return frames_tensor, mask
    overlay_frames = draw_pose_to_canvas_np(
        overlay_dw_pose_input,
        pool=None,
        H=output_height,
        W=output_width,
        reshape_scale=0,
        show_feet_flag=False,
        show_body_flag=False,
        show_cheek_flag=bool(draw_face),
        dw_hand=True,
        show_face_flag=bool(draw_face),
        show_hand_flag=bool(draw_hands),
    )
    if len(overlay_frames) != int(frames_tensor.shape[0]):
        logging.warning(
            "Render NLF Poses skipped DWPose 2D overlay: frame count mismatch "
            "rendered=%s overlay=%s",
            int(frames_tensor.shape[0]),
            len(overlay_frames),
        )
        return frames_tensor, mask
    overlay_tensor = torch.from_numpy(np.stack(overlay_frames, axis=0)).to(
        device=frames_tensor.device,
        dtype=frames_tensor.dtype,
    )
    if overlay_tensor.numel() == 0:
        return frames_tensor, mask
    overlay_tensor = overlay_tensor[..., :3] / 255.0
    active = (overlay_tensor > 0.0).any(dim=-1)
    if not bool(active.any().item()):
        return frames_tensor, mask

    result = frames_tensor.clone()
    result[active] = overlay_tensor[active]
    mask_result = torch.maximum(
        mask,
        active.to(device=mask.device, dtype=mask.dtype),
    )
    logging.info(
        "Render NLF Poses applied DWPose 2D overlay after geometry repair: "
        "frames=%s size=%sx%s draw_face=%s draw_hands=%s",
        int(frames_tensor.shape[0]),
        output_width,
        output_height,
        bool(draw_face),
        bool(draw_hands),
    )
    return result, mask_result


def _transform_normalized_points(points, transform, *, source_width, source_height):
    if points is None or transform is None or not transform.valid:
        return points
    array = np.array(points, copy=True)
    if array.shape[-1] < 2:
        return array
    width = float(source_width)
    height = float(source_height)
    if width <= 0.0 or height <= 0.0:
        return array
    x = array[..., 0]
    y = array[..., 1]
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (y >= 0.0)
    if not bool(np.any(valid)):
        return array
    transformed_x = (
        (x[valid] * width) * float(transform.scale_x) + float(transform.translate_x)
    ) / width
    transformed_y = (
        (y[valid] * height) * float(transform.scale_y) + float(transform.translate_y)
    ) / height
    array[..., 0][valid] = transformed_x
    array[..., 1][valid] = transformed_y
    return array


def _transform_dwpose_frame_coordinates(
    pose,
    *,
    transforms_by_person,
    source_width,
    source_height,
):
    if not transforms_by_person or source_width is None or source_height is None:
        return pose
    transformed = copy.deepcopy(pose)
    bodies = transformed.get("bodies", {})
    candidates = bodies.get("candidate")
    person_count = (
        int(candidates.shape[0])
        if hasattr(candidates, "shape") and candidates.ndim >= 3
        else 0
    )
    for person_index in range(person_count):
        transform = transforms_by_person.get(person_index)
        if transform is None:
            continue
        candidates[person_index] = _transform_normalized_points(
            candidates[person_index],
            transform,
            source_width=source_width,
            source_height=source_height,
        )
        faces = transformed.get("faces")
        if hasattr(faces, "shape") and faces.ndim >= 3 and person_index < faces.shape[0]:
            faces[person_index] = _transform_normalized_points(
                faces[person_index],
                transform,
                source_width=source_width,
                source_height=source_height,
            )
        hands = transformed.get("hands")
        if hasattr(hands, "shape") and hands.ndim >= 3:
            for hand_index in (person_index * 2, person_index * 2 + 1):
                if hand_index < hands.shape[0]:
                    hands[hand_index] = _transform_normalized_points(
                        hands[hand_index],
                        transform,
                        source_width=source_width,
                        source_height=source_height,
                    )
    return transformed


def _frame_transform(transforms, frame_index):
    if not transforms:
        return None
    if frame_index < len(transforms):
        return transforms[frame_index]
    if len(transforms) == 1:
        return transforms[0]
    return None


def _transform_dwpose_sequence_coordinates(
    dw_pose_input,
    *,
    transforms_by_person,
    source_width,
    source_height,
):
    if not transforms_by_person or source_width is None or source_height is None:
        return dw_pose_input
    transformed_frames = []
    for frame_index, pose in enumerate(dw_pose_input):
        frame_transforms = {
            int(person_index): transform
            for person_index, transforms in transforms_by_person.items()
            if (transform := _frame_transform(transforms, frame_index)) is not None
        }
        transformed_frames.append(
            _transform_dwpose_frame_coordinates(
                pose,
                transforms_by_person=frame_transforms,
                source_width=source_width,
                source_height=source_height,
            )
        )
    return transformed_frames


def _all_person_alignment_transforms(dw_pose_input, transforms):
    person_count = _dwpose_person_count(dw_pose_input)
    if person_count <= 0:
        person_count = 1
    return {person_index: transforms for person_index in range(person_count)}


def _dwpose_person_count(dw_pose_input):
    try:
        return int(dw_pose_input[0]["bodies"]["candidate"].shape[0])
    except Exception:
        return 0


def _pose_person_count(pose_input):
    for frame in pose_input:
        try:
            count = int(frame.shape[0])
            if count > 0:
                return count
        except Exception:
            try:
                count = len(frame)
                if count > 0:
                    return count
            except TypeError:
                continue
    return 0


def _slice_pose_input_person(pose_input, person_index):
    sliced = []
    for frame in pose_input:
        try:
            if int(frame.shape[0]) > person_index:
                sliced.append(frame[person_index : person_index + 1])
            else:
                sliced.append(frame[:0])
        except Exception:
            sliced.append([frame[person_index]] if len(frame) > person_index else [])
    return sliced


def _best_person_index_for_identity(
    *,
    normalized_bboxes,
    pose_video_mask,
    identity_index,
    person_count,
):
    if person_count <= 0:
        return None
    if normalized_bboxes.max_person_count <= 1:
        return identity_index if identity_index < person_count else None
    identity_mask = semantic_identity_rgb_mask(
        pose_video_mask,
        identity_index=identity_index,
    )
    target_boxes = frame_bboxes(identity_mask, kind="semantic_rgb_mask")
    best_index = None
    best_score = None
    for person_index in range(min(person_count, normalized_bboxes.max_person_count)):
        candidate = select_nlf_bboxes_for_identity(
            normalized_bboxes,
            identity_index=person_index,
        )
        distances = []
        for target_box, candidate_box in zip(target_boxes, candidate.boxes):
            if target_box is None or candidate_box is None:
                continue
            dx = target_box.center_x - candidate_box.center_x
            dy = target_box.center_y - candidate_box.center_y
            distances.append((dx * dx + dy * dy) ** 0.5)
        if not distances:
            continue
        score = sum(distances) / len(distances)
        if best_score is None or score < best_score:
            best_score = score
            best_index = person_index
    if best_index is None and identity_index < person_count:
        return identity_index
    return best_index


def _composite_identity_renders(rendered_tensors):
    _require_dependency("torch", torch)
    if not rendered_tensors:
        return None, None
    result = torch.zeros_like(rendered_tensors[0])
    mask = torch.zeros(
        rendered_tensors[0].shape[:3],
        device=rendered_tensors[0].device,
        dtype=torch.float32,
    )
    for tensor in rendered_tensors:
        active = (tensor[..., :3] > 0.001).any(dim=-1)
        result[active] = tensor[active]
        mask = torch.maximum(mask, active.to(device=mask.device, dtype=mask.dtype))
    return result, mask


def _nlf_frames_to_tensors(frames_np):
    _require_dependency("numpy", np)
    _require_dependency("torch", torch)
    frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
    return frames_tensor[..., :3].cpu().float(), (frames_tensor[..., -1] > 0.5).cpu().float()


def _format_nlf_detected_boxes(all_boxes):
    """Format detector xywh rows for the public BBOX output.

    Keep the legacy flat bbox shape for single-person frames, but preserve all
    candidates when the detector finds multiple people so RenderNLFPoses can
    match NLF person slots to semantic mask identities.
    """

    formatted = []
    for frame_boxes in all_boxes:
        if hasattr(frame_boxes, "detach"):
            frame_boxes = frame_boxes.detach()
        if hasattr(frame_boxes, "cpu"):
            frame_boxes = frame_boxes.cpu()
        if hasattr(frame_boxes, "numel") and frame_boxes.numel() == 0:
            formatted.append([0.0, 0.0, 0.0, 0.0])
            continue
        if hasattr(frame_boxes, "tolist"):
            rows = frame_boxes.tolist()
        else:
            rows = frame_boxes
        if not rows:
            formatted.append([0.0, 0.0, 0.0, 0.0])
            continue
        if isinstance(rows[0], (bool, int, float)):
            rows = [rows]
        candidates = []
        for row in rows:
            if len(row) < 4:
                continue
            x, y, width, height = (float(row[index]) for index in range(4))
            if width <= 0.0 or height <= 0.0:
                continue
            candidates.append([x, y, x + width, y + height])
        if not candidates:
            formatted.append([0.0, 0.0, 0.0, 0.0])
        elif len(candidates) == 1:
            formatted.append(candidates[0])
        else:
            formatted.append(candidates)
    return formatted


def _nlf_result_boxes_or_detector_boxes(result, detector_boxes):
    result_boxes = result.get("boxes") if isinstance(result, dict) else None
    if result_boxes is None:
        return detector_boxes
    # IMPORTANT: these boxes share the final filtering/order with poses3d.
    # Falling back to first-pass detector boxes here can pair a pose with the
    # wrong bbox and reintroduce RenderNLF scale/position drift.
    return result_boxes


_NLF_DETECTOR_MAX_BATCH_SIZE = 16


def _resolve_nlf_prediction_batch_size(num_images, per_batch):
    requested = int(per_batch)
    if requested == -1:
        return max(1, int(num_images))
    return max(1, requested)


def _resolve_nlf_detector_batch_size(num_images, per_batch):
    prediction_batch_size = _resolve_nlf_prediction_batch_size(num_images, per_batch)
    # IMPORTANT: ComfyUI RT-DETR can overflow in torch max_pool2d when long
    # videos are passed as one giant detector batch via per_batch=-1.
    return max(
        1,
        min(
            int(num_images) if int(num_images) > 0 else 1,
            prediction_batch_size,
            _NLF_DETECTOR_MAX_BATCH_SIZE,
        ),
    )


def _load_vitpose_utils():
    from .vitpose_utils.utils import (
        aaposemeta_to_dwpose_scail,
        bbox_from_detector,
        crop,
        load_pose_metas_from_kp2ds_seq,
    )

    return bbox_from_detector, crop, load_pose_metas_from_kp2ds_seq, aaposemeta_to_dwpose_scail


def _get_torch_device():
    return mm.get_torch_device() if mm is not None else "cpu"


def _get_offload_device():
    return mm.unet_offload_device() if mm is not None else "cpu"


def _resolve_taichi_render_device_key(render_device):
    requested = str(render_device or "gpu").lower()
    if requested == "gpu":
        cuda = getattr(torch, "cuda", None) if torch is not None else None
        if cuda is not None and cuda.is_available():
            return "cuda"
    return requested


def _initialize_taichi_backend(render_device):
    import taichi as ti

    resolved_key = _resolve_taichi_render_device_key(render_device)
    device_map = {
        "cpu": ti.cpu,
        "gpu": ti.gpu,
        "opengl": ti.opengl,
        "cuda": ti.cuda,
        "vulkan": ti.vulkan,
        "metal": ti.metal,
    }
    if resolved_key not in device_map:
        raise ValueError(f"Unsupported Taichi render device: {render_device!r}")
    ti.init(arch=device_map[resolved_key])
    try:
        active_arch = ti.lang.impl.current_cfg().arch
    except Exception:
        active_arch = "unknown"
    logging.info(
        "Render NLF Poses Taichi backend initialized: requested=%s "
        "resolved=%s active_arch=%s",
        render_device,
        resolved_key,
        active_arch,
    )
    return resolved_key, active_arch


device = _get_torch_device()
offload_device = _get_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))
folder_paths.add_model_folder_path("nlf", os.path.join(folder_paths.models_dir, "nlf"))

def convert_openpose_to_target_format(frames, max_people=2):
    _require_dependency("numpy", np)
    NUM_BODY = 18
    NUM_FACE = 70
    NUM_HAND = 21

    results = []
    for frame in frames:
        canvas_width = frame['canvas_width']
        canvas_height = frame['canvas_height']
        people = frame['people'][:max_people]

        bodies = []
        hands = []
        faces = []
        body_scores = []
        hand_scores = []
        face_scores = []

        for person in people:
            pose_raw = person.get('pose_keypoints_2d') or []
            if len(pose_raw) != NUM_BODY * 3:
                continue

            pose = np.array(pose_raw).reshape(-1, 3)
            pose_xy = np.stack([pose[:, 0] / canvas_width, pose[:, 1] / canvas_height], axis=1)
            bodies.append(pose_xy)
            body_scores.append(pose[:, 2])

            face_raw = person.get('face_keypoints_2d') or []
            if len(face_raw) == NUM_FACE * 3:
                face = np.array(face_raw).reshape(-1, 3)
                face_xy = np.stack([face[:, 0] / canvas_width, face[:, 1] / canvas_height], axis=1)
                faces.append(face_xy)
                face_scores.append(face[:, 2])

            hand_left_raw = person.get('hand_left_keypoints_2d') or []
            hand_right_raw = person.get('hand_right_keypoints_2d') or []
            if len(hand_left_raw) == NUM_HAND * 3:
                hand_left = np.array(hand_left_raw).reshape(-1, 3)
                hand_left_xy = np.stack([hand_left[:, 0] / canvas_width, hand_left[:, 1] / canvas_height], axis=1)
                hands.append(hand_left_xy)
                hand_scores.append(hand_left[:, 2])
            if len(hand_right_raw) == NUM_HAND * 3:
                hand_right = np.array(hand_right_raw).reshape(-1, 3)
                hand_right_xy = np.stack([hand_right[:, 0] / canvas_width, hand_right[:, 1] / canvas_height], axis=1)
                hands.append(hand_right_xy)
                hand_scores.append(hand_right[:, 2])

        result = {
            'bodies': {
                'candidate': np.array(bodies, dtype=np.float32),
                'subset': np.array([np.arange(NUM_BODY) for _ in bodies], dtype=np.float32) if bodies else np.array([])
            },
            'hands': np.array(hands, dtype=np.float32),
            'faces': np.array(faces, dtype=np.float32),
            'body_score': np.array(body_scores, dtype=np.float32),
            'hand_score': np.array(hand_scores, dtype=np.float32),
            'face_score': np.array(face_scores, dtype=np.float32)
        }
        results.append(result)
    return results

def scale_faces(poses, pose_2d_ref):
    _require_dependency("numpy", np)
    # Input: two lists of dict, poses[0]['faces'].shape: 1, 68, 2  , poses_ref[0]['faces'].shape: 1, 68, 2
    # Scale the facial keypoints in poses according to the center point of the face
    # That is: calculate the distance from the center point (idx: 30) to other facial keypoints in ref,
    # and the same for poses, then get scale_n as the ratio
    # Clamp scale_n to the range 0.8-1.5, then apply it to poses
    # Note: poses are modified in place

    ref = pose_2d_ref[0]
    pose_0 = poses[0]

    face_0 = pose_0['faces']  # shape: (1, 68, 2)
    face_ref = ref['faces']

    # Extract numpy arrays
    face_0 = np.array(face_0[0])      # (68, 2)
    face_ref = np.array(face_ref[0])

    # Center point (nose tip or face center)
    center_idx = 30
    center_0 = face_0[center_idx]
    center_ref = face_ref[center_idx]

    # Calculate distance to center point
    dist = np.linalg.norm(face_0 - center_0, axis=1)
    dist_ref = np.linalg.norm(face_ref - center_ref, axis=1)

    # Avoid the 0 distance of the center point itself
    dist = np.delete(dist, center_idx)
    dist_ref = np.delete(dist_ref, center_idx)

    mean_dist = np.mean(dist)
    mean_dist_ref = np.mean(dist_ref)

    if mean_dist < 1e-6:
        scale_n = 1.0
    else:
        scale_n = mean_dist_ref / mean_dist

    # Clamp to [0.8, 1.5]
    scale_n = np.clip(scale_n, 0.8, 1.5)

    for i, pose in enumerate(poses):
        face = pose['faces']
        # Extract numpy array
        face = np.array(face[0])      # (68, 2)
        center = face[center_idx]
        scaled_face = (face - center) * scale_n + center
        poses[i]['faces'][0] = scaled_face

        body = pose['bodies']
        candidate = body['candidate']
        candidate_np = np.array(candidate[0])   # (14, 2)
        body_center = candidate_np[0]
        scaled_candidate = (candidate_np - body_center) * scale_n + body_center
        poses[i]['bodies']['candidate'][0] = scaled_candidate

    # In-place modification
    pose['faces'][0] = scaled_face

    return scale_n

class PoseDetectionVitPoseToDWPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": ("POSEMODEL",),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "ViTPose to DWPose format pose detection node."

    def process(self, vitpose_model, images):
        _require_dependency("numpy", np)
        _require_dependency("cv2", cv2)
        bbox_from_detector, crop, load_pose_metas_from_kp2ds_seq, aaposemeta_to_dwpose_scail = _load_vitpose_utils()

        detector = vitpose_model["yolo"]
        pose_model = vitpose_model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        comfy_pbar = ProgressBar(B*2)
        progress = 0
        bboxes = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting bboxes"):
            bboxes.append(detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape
                )[0][0]["bbox"])
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        kp2ds = []
        for img, bbox in tqdm(zip(images_np, bboxes), total=len(images_np), desc="Extracting keypoints"):
            if bbox is None or bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                bbox = np.array([0, 0, img.shape[1], img.shape[0]])

            bbox_xywh = bbox
            center, scale = bbox_from_detector(bbox_xywh, input_resolution, rescale=rescale)
            img = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

            img_norm = (img - IMG_NORM_MEAN) / IMG_NORM_STD
            img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

            keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
            kp2ds.append(keypoints)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        kp2ds = np.concatenate(kp2ds, 0)
        pose_metas = load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)
        dwposes = [aaposemeta_to_dwpose_scail(meta) for meta in pose_metas]
        swap_hands = True
        out_dict = {"poses": dwposes, "swap_hands": swap_hands}
        return out_dict,


class ConvertOpenPoseKeypointsToDWPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keypoints": ("POSE_KEYPOINT",),
                "max_people": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1, "tooltip": "Maximum number of people to process per frame"}),
            },
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Convert OpenPose format keypoints to DWPose format."

    def process(self, keypoints, max_people=2):
        swap_hands = False
        out_dict = {"poses": convert_openpose_to_target_format(keypoints, max_people=max_people), "swap_hands": swap_hands}
        return out_dict,


class RenderNLFPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "nlf_poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "render_width": ("INT", {"default": 1024, "min": 2, "max": 8192, "step": 8, "tooltip": "Source render width. Output pose video width is render_width / 2."}),
            "render_height": ("INT", {"default": 1024, "min": 2, "max": 8192, "step": 8, "tooltip": "Source render height. Output pose video height is render_height / 2."}),
            },
            "optional": {
                "dw_poses": ("DWPOSES", {"default": None, "tooltip": "Optional DW pose model for 2D drawing"}),
                "ref_dw_pose": ("DWPOSES", {"default": None, "tooltip": "Optional reference DW pose model for alignment"}),
                "bboxes": ("BBOX", {"default": None, "tooltip": "Optional NLF Predict bbox output used to diagnose and repair render geometry"}),
                "pose_video_mask": ("IMAGE", {"default": None, "tooltip": "Optional SAM3/SCAIL-2 pose_video_mask used to align rendered pose geometry"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw face keypoints"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw hand keypoints"}),
                "render_device": (["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"], {"default": "gpu", "tooltip": "Taichi device to use for rendering"}),
                "scale_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to scale hand keypoints when aligning DW poses"}),
                "render_backend": (["taichi", "torch"], {"default": "taichi", "tooltip": "Rendering backend to use"}),
            }
    }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "predict"
    CATEGORY = "WanVideoWrapper"

    def predict(
        self,
        nlf_poses,
        render_width,
        render_height,
        dw_poses=None,
        ref_dw_pose=None,
        bboxes=None,
        pose_video_mask=None,
        draw_face=True,
        draw_hands=True,
        render_device="gpu",
        scale_hands=True,
        render_backend="taichi",
    ):
        _require_dependency("numpy", np)
        _require_dependency("torch", torch)

        from .NLFPoseExtract import nlf_render as nlf_render_module
        from .NLFPoseExtract.align3d import solve_new_camera_params_central, solve_new_camera_params_down
        render_nlf_as_images = nlf_render_module.render_nlf_as_images
        render_multi_nlf_as_images = nlf_render_module.render_multi_nlf_as_images
        shift_dwpose_according_to_nlf = nlf_render_module.shift_dwpose_according_to_nlf
        process_data_to_COCO_format = nlf_render_module.process_data_to_COCO_format
        intrinsic_matrix_from_field_of_view = nlf_render_module.intrinsic_matrix_from_field_of_view
        if render_backend == "taichi":
            if getattr(nlf_render_module, "render_whole_taichi", None) is None:
                logging.warning(
                    "Render NLF Poses Taichi renderer is unavailable; "
                    "falling back to torch rendering."
                )
                render_backend = "torch"
            else:
                try:
                    _initialize_taichi_backend(render_device)
                except Exception as exc:
                    logging.warning(
                        "Render NLF Poses Taichi init failed for "
                        "render_device=%s; falling back to torch rendering. "
                        "error=%s",
                        render_device,
                        exc,
                    )
                    render_backend = "torch"
        logging.info(
            "Render NLF Poses render backend selected: backend=%s device=%s",
            render_backend,
            render_device,
        )
        logging.info(
            "Render NLF Poses runtime provenance: %s",
            _render_nlf_runtime_provenance(),
        )

        if isinstance(nlf_poses, dict):
            pose_input = nlf_poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses else nlf_poses
        else:
            pose_input = nlf_poses

        active_render_width = int(render_width)
        active_render_height = int(render_height)
        if active_render_width <= 0 or active_render_height <= 0:
            raise ValueError("render_width and render_height must be positive")
        output_width = max(active_render_width // 2, 1)
        output_height = max(active_render_height // 2, 1)

        dw_pose_input = copy.deepcopy(dw_poses["poses"]) if dw_poses is not None else None
        swap_hands = dw_poses.get("swap_hands", False) if dw_poses is not None else False

        ori_camera_pose = intrinsic_matrix_from_field_of_view([active_render_height, active_render_width])
        ori_focal = ori_camera_pose[0, 0]

        num_people = dw_pose_input[0]['bodies']['candidate'].shape[0] if dw_poses is not None else 0

        if dw_poses is not None and ref_dw_pose is not None and num_people == 1:
            ref_dw_pose_input = copy.deepcopy(ref_dw_pose["poses"])

            # Find the first valid pose
            pose_3d_first_driving_frame = None
            for pose in pose_input:
                if pose.shape[0] == 0:
                    continue
                candidate = pose[0].cpu().numpy()
                if np.any(candidate):
                    pose_3d_first_driving_frame = candidate
                    break
            if pose_3d_first_driving_frame is None:
                raise ValueError("No valid pose found in pose_input.")

            pose_3d_coco_first_driving_frame = process_data_to_COCO_format(pose_3d_first_driving_frame)
            poses_2d_ref = ref_dw_pose_input[0]['bodies']['candidate'][0][:14]
            poses_2d_ref[:, 0] = poses_2d_ref[:, 0] * active_render_width
            poses_2d_ref[:, 1] = poses_2d_ref[:, 1] * active_render_height

            poses_2d_subset = ref_dw_pose_input[0]['bodies']['subset'][0][:14]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[:14]

            valid_indices, valid_upper_indices, valid_lower_indices = [], [], []
            upper_body_indices = [0, 2, 3, 5, 6]
            lower_body_indices = [9, 10, 12, 13]

            for i in range(len(poses_2d_subset)):
                if poses_2d_subset[i] != -1.0 and np.sum(pose_3d_coco_first_driving_frame[i]) != 0:
                    if i in upper_body_indices:
                        valid_upper_indices.append(i)
                    if i in lower_body_indices:
                        valid_lower_indices.append(i)

            valid_indices = [1] + valid_lower_indices if len(valid_upper_indices) < 4 else [1] + valid_lower_indices + valid_upper_indices # align body or only lower body

            pose_2d_ref = poses_2d_ref[valid_indices]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[valid_indices]

            if len(valid_lower_indices) >= 4:
                camera_solve_mode = "down"
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_down(pose_3d_coco_first_driving_frame, ori_focal, [active_render_height, active_render_width], pose_2d_ref)
            else:
                camera_solve_mode = "central"
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_central(pose_3d_coco_first_driving_frame, ori_focal, [active_render_height, active_render_width], pose_2d_ref)

            camera_solve_validation = validate_ref_dwpose_camera_solve(
                camera_intrinsics=new_camera_intrinsics,
                scale_m=scale_m,
                scale_s=scale_s,
                points_3d=pose_3d_coco_first_driving_frame,
                target_points_2d=pose_2d_ref,
                width=active_render_width,
                height=active_render_height,
                solve_mode=camera_solve_mode,
            )
            if camera_solve_validation.safe:
                scale_face = scale_faces(list(dw_pose_input), list(ref_dw_pose_input))   # poses[0]['faces'].shape: 1, 68, 2  , poses_ref[0]['faces'].shape: 1, 68, 2
                logging.info(
                    "Render NLF Poses ref_dw_pose camera solve: %s face_scale=%s",
                    camera_solve_validation.summary,
                    scale_face,
                )
                shift_dwpose_according_to_nlf(pose_input, dw_pose_input, ori_camera_pose, new_camera_intrinsics, active_render_height, active_render_width, swap_hands=swap_hands, scale_hands=scale_hands, scale_x=scale_m, scale_y=scale_m*scale_s)
                intrinsic_matrix = new_camera_intrinsics
            else:
                logging.warning(
                    "Render NLF Poses ref_dw_pose camera solve rejected: %s",
                    camera_solve_validation.summary,
                )
                intrinsic_matrix = ori_camera_pose
        else:
            intrinsic_matrix = ori_camera_pose

        normalized_bboxes = normalize_nlf_bboxes(bboxes, frame_count=len(pose_input))
        logging.info(
            "Render NLF Poses source canvas: %s",
            format_nlf_source_canvas_diagnostics(
                render_width=active_render_width,
                render_height=active_render_height,
                output_width=output_width,
                output_height=output_height,
                pose_video_mask=pose_video_mask,
                normalized_bboxes=normalized_bboxes,
                bboxes_connected=bboxes is not None,
                dw_pose_input=dw_pose_input,
            ),
        )
        if bboxes is not None:
            logging.info("Render NLF Poses bbox diagnostics: %s", normalized_bboxes.summary())

        pose_person_count = _pose_person_count(pose_input)
        semantic_identity_count = (
            identity_count_from_semantic_mask(pose_video_mask)
            if pose_video_mask is not None
            else 0
        )
        logging.info(
            "Render NLF Poses geometry inputs: pose_persons=%s "
            "semantic_identities=%s bbox_max_persons=%s bbox_valid=%s",
            pose_person_count,
            semantic_identity_count,
            normalized_bboxes.max_person_count,
            normalized_bboxes.valid_count,
        )
        used_identity_composition = False
        dwpose_alignment_transforms_by_person = {}
        frames_tensor = None
        mask = None
        if (
            pose_video_mask is not None
            and semantic_identity_count > 0
            and pose_person_count > 1
        ):
            if normalized_bboxes.max_person_count <= 1:
                logging.warning(
                    "Render NLF Poses identity matching has no multi-person "
                    "bbox candidates; falling back to NLF/SAM index order. "
                    "Connect NLFPredictPoses.bboxes to RenderNLFPoses.bboxes "
                    "and restart after updating the node pack."
                )
            rendered_identity_tensors = []
            used_person_indices = []
            for identity_index in range(min(semantic_identity_count, pose_person_count)):
                person_index = _best_person_index_for_identity(
                    normalized_bboxes=normalized_bboxes,
                    pose_video_mask=pose_video_mask,
                    identity_index=identity_index,
                    person_count=pose_person_count,
                )
                if person_index is None:
                    logging.warning(
                        "Render NLF Poses skipped identity render: "
                        "identity=%s person_count=%s",
                        identity_index,
                        pose_person_count,
                    )
                    continue
                identity_pose_input = _slice_pose_input_person(pose_input, person_index)
                identity_frames_np = render_nlf_as_images(
                    identity_pose_input,
                    None,
                    active_render_height,
                    active_render_width,
                    len(pose_input),
                    intrinsic_matrix=intrinsic_matrix,
                    draw_face=False,
                    draw_hands=False,
                    render_backend=render_backend,
                )
                identity_tensor, _identity_mask = _nlf_frames_to_tensors(identity_frames_np)
                identity_semantic_mask = semantic_identity_rgb_mask(
                    pose_video_mask,
                    identity_index=identity_index,
                )
                identity_alignment = align_pose_video_to_mask(
                    pose_video=identity_tensor,
                    pose_video_mask=identity_semantic_mask,
                )
                rendered_identity_tensors.append(identity_alignment.pose_video.cpu().float())
                if identity_alignment.alignment_transforms:
                    dwpose_alignment_transforms_by_person[person_index] = (
                        identity_alignment.alignment_transforms
                    )
                used_person_indices.append(person_index)
                logging.info(
                    "Render NLF Poses identity alignment: identity=%s person=%s %s",
                    identity_index,
                    person_index,
                    identity_alignment.summary,
                )

            composed_frames, composed_mask = _composite_identity_renders(
                rendered_identity_tensors
            )
            if composed_frames is not None and composed_mask is not None:
                frames_tensor = composed_frames.cpu().float()
                mask = composed_mask.cpu().float()
                used_identity_composition = True
                logging.info(
                    "Render NLF Poses identity composition: identities=%s "
                    "pose_persons=%s used_person_indices=%s",
                    semantic_identity_count,
                    pose_person_count,
                    used_person_indices,
                )
            else:
                logging.warning(
                    "Render NLF Poses identity composition produced no frames; "
                    "falling back to legacy render path"
                )

        if frames_tensor is None or mask is None:
            if pose_person_count > 1:
                frames_np = render_multi_nlf_as_images(
                    pose_input,
                    None,
                    active_render_height,
                    active_render_width,
                    len(pose_input),
                    intrinsic_matrix=intrinsic_matrix,
                    draw_face=False,
                    draw_hands=False,
                    render_backend=render_backend,
                )
            else:
                frames_np = render_nlf_as_images(
                    pose_input,
                    None,
                    active_render_height,
                    active_render_width,
                    len(pose_input),
                    intrinsic_matrix=intrinsic_matrix,
                    draw_face=False,
                    draw_hands=False,
                    render_backend=render_backend,
                )
            frames_tensor, mask = _nlf_frames_to_tensors(frames_np)

        mask_alignment_applied = False
        mask_alignment_skipped_reason = None
        if pose_video_mask is not None and not used_identity_composition:
            try:
                alignment = align_pose_video_to_mask(
                    pose_video=frames_tensor,
                    pose_video_mask=pose_video_mask,
                )
                mask_alignment_safe, mask_alignment_reason = (
                    pose_mask_alignment_is_safe_for_render_repair(
                        alignment,
                        normalized_bboxes=normalized_bboxes,
                        bboxes_connected=bboxes is not None,
                        width=active_render_width,
                        height=active_render_height,
                    )
                )
            except ValueError as exc:
                if "frame counts must match" in str(exc):
                    mask_alignment_skipped_reason = "mask_frame_count_mismatch"
                else:
                    mask_alignment_skipped_reason = f"mask_alignment_error:{type(exc).__name__}"
                logging.warning(
                    "Render NLF Poses pose/mask alignment skipped: %s",
                    mask_alignment_skipped_reason,
                )
            else:
                if mask_alignment_safe:
                    frames_tensor = alignment.pose_video.cpu().float()
                    mask = (frames_tensor[..., :3] > 0.001).any(dim=-1).float()
                    if alignment.alignment_transforms:
                        dwpose_alignment_transforms_by_person = _all_person_alignment_transforms(
                            dw_pose_input,
                            alignment.alignment_transforms,
                        )
                    mask_alignment_applied = True
                    logging.info("Render NLF Poses pose/mask alignment: %s", alignment.summary)
                else:
                    mask_alignment_skipped_reason = mask_alignment_reason
                    logging.warning(
                        "Render NLF Poses pose/mask alignment skipped: %s",
                        mask_alignment_skipped_reason,
                    )

        if (
            not mask_alignment_applied
            and bboxes is not None
            and not used_identity_composition
        ):
            can_repair, reason = bbox_payload_is_safe_for_render_repair(
                normalized_bboxes,
                width=active_render_width,
                height=active_render_height,
            )
            fallback_reason = "none" if can_repair else reason
            if mask_alignment_skipped_reason is not None:
                fallback_reason = (
                    f"{mask_alignment_skipped_reason}->bbox"
                    if can_repair
                    else f"{mask_alignment_skipped_reason};{reason}"
                )
            logging.info(
                "Render NLF Poses bbox geometry: %s",
                format_nlf_render_bbox_diagnostics(
                    pose_video=frames_tensor,
                    target_bboxes=normalized_bboxes.boxes,
                    target_source="nlf_bboxes",
                    width=active_render_width,
                    height=active_render_height,
                    fallback_reason=fallback_reason,
                ),
            )
            if can_repair:
                alignment = align_pose_video_to_bboxes(
                    pose_video=frames_tensor,
                    bboxes=normalized_bboxes.boxes,
                )
                frames_tensor = alignment.pose_video.cpu().float()
                mask = (frames_tensor[..., :3] > 0.001).any(dim=-1).float()
                if alignment.alignment_transforms:
                    dwpose_alignment_transforms_by_person = _all_person_alignment_transforms(
                        dw_pose_input,
                        alignment.alignment_transforms,
                    )
                logging.info("Render NLF Poses bbox alignment: %s", alignment.summary)
            else:
                logging.warning("Render NLF Poses bbox alignment skipped: %s", reason)

        frames_tensor = resize_bhwc_video(
            frames_tensor,
            width=output_width,
            height=output_height,
        ).cpu().float()
        mask = resize_mask_video(mask, width=output_width, height=output_height).cpu().float()
        logging.info(
            "Render NLF Poses downsampled render %sx%s to half-size output %sx%s",
            active_render_width,
            active_render_height,
            output_width,
            output_height,
        )
        frames_tensor, mask = _overlay_dwpose_2d_on_frames(
            frames_tensor=frames_tensor,
            mask=mask,
            dw_pose_input=dw_pose_input,
            draw_face=draw_face,
            draw_hands=draw_hands,
            identity_count=semantic_identity_count if semantic_identity_count > 0 else None,
            alignment_transforms_by_person=dwpose_alignment_transforms_by_person,
            alignment_source_width=active_render_width,
            alignment_source_height=active_render_height,
        )

        return (frames_tensor, mask)

class SaveNLFPosesAs3D:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "nlf_poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "filename_prefix": ("STRING", {"default": "nlf_pose_3d"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 300.0, "step": 0.1, "tooltip": "Frames per second for the output animation"}),
            "cylinder_radius": ("FLOAT", {"default": 21.5, "tooltip": "Radius of the cylinders representing bones"}),
            },
    }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True
    FUNCTION = "save_3d"
    CATEGORY = "WanVideoWrapper"

    def save_3d(self, nlf_poses, filename_prefix, fps, cylinder_radius):
        from .NLFPoseExtract.nlf_render import get_cylinder_specs_list_from_poses
        from .render_3d.export_utils import save_cylinder_specs_as_glb_animation
        try:
            if isinstance(nlf_poses, dict):
                pose_input = nlf_poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses else nlf_poses
            else:
                pose_input = nlf_poses

            cylinder_specs_list = get_cylinder_specs_list_from_poses(pose_input, include_missing=True)
            logging.info(f"Generated {len(cylinder_specs_list)} frames of cylinder specs")

            output_dir = folder_paths.get_output_directory()
            full_output_folder = os.path.join(output_dir, filename_prefix)
            if not os.path.exists(full_output_folder):
                os.makedirs(full_output_folder)

            filename = f"{filename_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"
            filepath = os.path.join(full_output_folder, filename)

            logging.info(f"Saving as GLB animation to {full_output_folder}")
            logging.info(f"Starting GLB animation export. Frames: {len(cylinder_specs_list)}")
            save_cylinder_specs_as_glb_animation(cylinder_specs_list, filepath, fps=fps, radius=cylinder_radius)
            logging.info(f"Saved GLB: {filepath}")
        except Exception as e:
            logging.error(f"Error in SaveNLFPosesAs3D: {e}")
            raise e

        return (filepath,)


# NLF model loader

class NLFModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_model": (folder_paths.get_filename_list("nlf"), {
                    "tooltip": "NLF model (.safetensors) from ComfyUI/models/nlf/",
                }),
            },
        }

    RETURN_TYPES = ("NLF_MODEL",)
    RETURN_NAMES = ("nlf_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "SCAIL-Pose"

    def loadmodel(self, nlf_model):
        _require_dependency("comfy", comfy)
        _require_dependency("comfy.model_management", mm)
        from safetensors.torch import load_file
        from .nlf_model.model import NLFModel
        from .nlf_model.multiperson import MultipersonNLF, load_detector

        model_path = folder_paths.get_full_path_or_raise("nlf", nlf_model)
        logging.info(f"Loading NLF model from {model_path}")
        sd = load_file(model_path)

        crop_sd = {}
        detector_sd = {}
        for k, v in sd.items():
            if k.startswith('detector.'):
                detector_sd[k[len('detector.'):]] = v
            elif not k.startswith('cano_all.'):
                crop_sd[k] = v

        crop_model = NLFModel.from_state_dict(crop_sd, operations=comfy.ops.manual_cast).eval()

        # Wrap in ModelPatcher for ComfyUI memory management
        load_device = mm.get_torch_device()
        model_patcher = comfy.model_patcher.ModelPatcher(
            crop_model, load_device=load_device, offload_device=offload_device
        )

        detector = None
        if detector_sd:
            logging.info(f"Loading bundled RT-DETR detector ({len(detector_sd)} keys)")
            detector = load_detector(detector_sd)

        # SMPL canonical points: first 1024 are vertices, last 24 are joints
        canonical_points = sd.get('cano_all.smpl', crop_model.canonical_locs())
        num_vertices = 1024 if 'cano_all.smpl' in sd else 0

        pipeline = MultipersonNLF(
            crop_model=crop_model,
            model_patcher=model_patcher,
            detector=detector,
            canonical_points=canonical_points,
            num_vertices=num_vertices,
        )

        logging.info("NLF model loaded successfully")
        return (pipeline,)


class NLFPredictPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "nlf_model": ("NLF_MODEL",),
                "images": ("IMAGE", {"tooltip": "Input images (BHWC format)"}),
            },
            "optional": {
                "per_batch": ("INT", {"default": 1, "min": -1, "max": 10000, "step": 1,
                    "tooltip": "Images per pose-estimation batch. -1 = all at once for NLF estimation; person detection is automatically chunked for long videos. 1 = lowest VRAM usage."}),
                "num_aug": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Number of test-time augmentations. More = slower but more accurate."}),
                "detector_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Person detection confidence threshold"}),
            },
        }

    RETURN_TYPES = ("NLFPRED", "BBOX",)
    RETURN_NAMES = ("pose_results", "bboxes")
    FUNCTION = "predict"
    CATEGORY = "SCAIL-Pose"

    def predict(self, nlf_model, images, per_batch=1, num_aug=1, detector_threshold=0.3):
        _require_dependency("torch", torch)
        _require_dependency("comfy.model_management", mm)
        num_images = images.shape[0]
        detector_batch_size = _resolve_nlf_detector_batch_size(num_images, per_batch)
        pose_batch_size = _resolve_nlf_prediction_batch_size(num_images, per_batch)
        if detector_batch_size != pose_batch_size:
            logging.info(
                "NLF Predict Poses detector batch bounded: frames=%s "
                "pose_batch=%s detector_batch=%s",
                num_images,
                pose_batch_size,
                detector_batch_size,
            )

        # Convert to NCHW on GPU once
        images_nchw = images.permute(0, 3, 1, 2)

        # Phase 1: Detect persons per frame
        nlf_model.detector.load()
        all_boxes = []
        pbar_det = ProgressBar(num_images)
        for i in tqdm(range(0, num_images, detector_batch_size), desc="Detecting"):
            end_idx = min(i + detector_batch_size, num_images)
            all_boxes.extend(nlf_model.detector.detect(images_nchw[i:end_idx].to(device), threshold=detector_threshold))
            pbar_det.update(end_idx - i)

        # Phase 2: Estimate poses per frame
        mm.load_model_gpu(nlf_model.model_patcher)
        all_joints3d = []
        all_result_boxes = []
        pbar_nlf = ProgressBar(num_images)
        for i in tqdm(range(0, num_images, pose_batch_size), desc="Estimating poses"):
            end_idx = min(i + pose_batch_size, num_images)
            detector_batch_boxes = all_boxes[i:end_idx]

            result = nlf_model.detect_and_estimate(
                images_nchw[i:end_idx].to(device), num_aug=num_aug, boxes=detector_batch_boxes,
            )

            all_joints3d.extend(result['poses3d'])
            all_result_boxes.extend(
                _nlf_result_boxes_or_detector_boxes(result, detector_batch_boxes)
            )
            pbar_nlf.update(end_idx - i)

        all_result_boxes = [b.to(offload_device) for b in all_result_boxes]
        all_joints3d = [j.to(offload_device) for j in all_joints3d]

        pose_results = {
            'joints3d_nonparam': [all_joints3d],
        }

        formatted_boxes = _format_nlf_detected_boxes(all_result_boxes)

        return (pose_results, formatted_boxes)


from .nodes_wanvideo_scail2_adapter import (  # noqa: E402
    NODE_CLASS_MAPPINGS as WANVIDEO_SCAIL2_ADAPTER_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as WANVIDEO_SCAIL2_ADAPTER_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_sam3_preprocessing import (  # noqa: E402
    NODE_CLASS_MAPPINGS as SAM3_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SAM3_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_scail2_condition import (  # noqa: E402
    NODE_CLASS_MAPPINGS as SCAIL2_CONDITION_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SCAIL2_CONDITION_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_scail2_replacement_mask import (  # noqa: E402
    NODE_CLASS_MAPPINGS as SCAIL2_REPLACEMENT_MASK_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SCAIL2_REPLACEMENT_MASK_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_scail2_replacement_condition_video import (  # noqa: E402
    NODE_CLASS_MAPPINGS as SCAIL2_REPLACEMENT_CONDITION_VIDEO_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SCAIL2_REPLACEMENT_CONDITION_VIDEO_NODE_DISPLAY_NAME_MAPPINGS,
)
from .nodes_scail2_pose_alignment import (  # noqa: E402
    NODE_CLASS_MAPPINGS as SCAIL2_POSE_ALIGNMENT_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as SCAIL2_POSE_ALIGNMENT_NODE_DISPLAY_NAME_MAPPINGS,
)


NODE_CLASS_MAPPINGS = {
    "PoseDetectionVitPoseToDWPose": PoseDetectionVitPoseToDWPose,
    "RenderNLFPoses": RenderNLFPoses,
    "ConvertOpenPoseKeypointsToDWPose": ConvertOpenPoseKeypointsToDWPose,
    "SaveNLFPosesAs3D": SaveNLFPosesAs3D,
    "NLFModelLoader": NLFModelLoader,
    "NLFPredictPoses": NLFPredictPoses,
}
NODE_CLASS_MAPPINGS.update(WANVIDEO_SCAIL2_ADAPTER_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SAM3_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SCAIL2_CONDITION_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SCAIL2_REPLACEMENT_MASK_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SCAIL2_REPLACEMENT_CONDITION_VIDEO_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(SCAIL2_POSE_ALIGNMENT_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseDetectionVitPoseToDWPose": "Pose Detection VitPose to DWPose",
    "RenderNLFPoses": "Render NLF Poses",
    "ConvertOpenPoseKeypointsToDWPose": "Convert OpenPose Keypoints to DWPose",
    "SaveNLFPosesAs3D": "Save NLF Poses as 3D Animation",
    "NLFModelLoader": "NLF Model Loader",
    "NLFPredictPoses": "NLF Predict Poses",
}
NODE_DISPLAY_NAME_MAPPINGS.update(WANVIDEO_SCAIL2_ADAPTER_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SAM3_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SCAIL2_CONDITION_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SCAIL2_REPLACEMENT_MASK_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SCAIL2_REPLACEMENT_CONDITION_VIDEO_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SCAIL2_POSE_ALIGNMENT_NODE_DISPLAY_NAME_MAPPINGS)
