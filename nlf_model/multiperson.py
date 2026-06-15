"""Multi-person NLF pose estimation pipeline + person detector."""

import math

import torch
import torch.nn.functional as F
import torchvision.ops

import comfy.model_management as mm
import comfy.sd

from .model import _linspace, to_homogeneous, intrinsic_matrix_from_field_of_view, lookat_matrix, rotation_mat

# Person detector (RT-DETR via ComfyUI core)

PERSON_CLASS_ID = 0  # COCO class 0 = 'person'


class PersonDetector:
    """Wrapper around a ComfyUI-loaded RT-DETR model for person detection."""

    def __init__(self, model_patcher):
        self.model_patcher = model_patcher

    def load(self):
        mm.load_model_gpu(self.model_patcher)

    def detect(self, images, threshold=0.3, nms_iou_threshold=0.7, max_detections=150):
        device = mm.get_torch_device()
        model = self.model_patcher.model.diffusion_model
        dtype = model.dtype if hasattr(model, 'dtype') else torch.float32

        orig_h, orig_w = images.shape[2], images.shape[3]
        resized = F.interpolate(images, size=(640, 640), mode='bilinear', align_corners=False)
        results = model(resized.to(device=device, dtype=dtype), orig_size=(orig_w, orig_h))

        boxes_list = []
        for det in results:
            person_mask = det['labels'] == PERSON_CLASS_ID
            boxes_xyxy = det['boxes'][person_mask]
            scores = det['scores'][person_mask]
            conf_mask = scores >= threshold
            boxes_xyxy, scores = boxes_xyxy[conf_mask], scores[conf_mask]

            if len(boxes_xyxy) > 0:
                keep = torchvision.ops.nms(boxes_xyxy, scores, nms_iou_threshold)
                if max_detections > 0:
                    keep = keep[:max_detections]
                boxes_xyxy, scores = boxes_xyxy[keep], scores[keep]
                x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
                boxes_list.append(mm.cast_to(torch.stack([x1, y1, x2 - x1, y2 - y1, scores], dim=-1), dtype=torch.float32))
            else:
                boxes_list.append(torch.zeros((0, 5), dtype=torch.float32, device=device))
        return boxes_list


def load_detector(detector_sd):
    model_patcher = comfy.sd.load_diffusion_model_state_dict(detector_sd)
    if model_patcher is None:
        return None
    return PersonDetector(model_patcher)


# Warping

def _corner_aligned_scale_mat(factor, device=None):
    s = factor
    return torch.tensor([[s, 0, (s-1)/2], [0, s, (s-1)/2], [0, 0, 1]], dtype=torch.float32, device=device)


def _project(points):
    return points[..., :2] / points[..., 2:3]


_grid_cache = {}

def _get_homogeneous_grid(oh, ow, device):
    """Cached [oh*ow, 3] homogeneous pixel coordinate grid."""
    key = (oh, ow, device)
    if key not in _grid_cache:
        gy, gx = torch.meshgrid(torch.arange(oh, device=device), torch.arange(ow, device=device), indexing='ij')
        _grid_cache[key] = torch.stack([gx.reshape(-1), gy.reshape(-1), torch.ones(oh * ow, device=device)], dim=-1).float()
    return _grid_cache[key]


def _warp_images_pyramid(images, intrinsic_matrix, new_invprojmats, crop_scales, output_shape, image_ids, n_levels=3):
    device = images.device
    n_crops = len(image_ids)
    oh, ow = output_shape

    # Build image pyramid
    levels = [images]
    for _ in range(1, n_levels):
        levels.append(F.avg_pool2d(levels[-1], 2, 2))
    intr_levels = [_corner_aligned_scale_mat(1 / 2 ** i, device=device) @ intrinsic_matrix for i in range(n_levels)]
    pyr = torch.clip(torch.floor(-torch.log2(crop_scales)), 0, n_levels - 1).int()

    # Full homography per crop: H = intrinsic_at_level @ inv_proj
    per_crop_intr = torch.stack([intr_levels[pyr[i]][i] for i in range(n_crops)])
    homographies = per_crop_intr @ new_invprojmats  # [n_crops, 3, 3]

    # Batched homography application + perspective divide
    grid_flat = _get_homogeneous_grid(oh, ow, device)
    old_coords = torch.einsum('nij,pj->npi', homographies, grid_flat)
    old_2d = (old_coords[..., :2] / old_coords[..., 2:3]).reshape(n_crops, oh, ow, 2)

    # grid_sample per pyramid level (crops within a level share source image size)
    result = torch.empty(n_crops, images.shape[1], oh, ow, device=device, dtype=images.dtype)
    for lvl in range(n_levels):
        mask = pyr == lvl
        if not mask.any():
            continue
        idx = torch.where(mask)[0]
        src = levels[lvl][image_ids[idx]]
        size = torch.tensor([src.shape[3], src.shape[2]], dtype=torch.float32, device=device)
        grid = old_2d[idx] * (2.0 / (size - 1)) - 1.0
        result[idx] = F.grid_sample(
            src, grid.to(src.dtype),
            align_corners=True, mode='bilinear', padding_mode='zeros',
        )

    return result


# Plausibility / NMS / Aggregation

def _is_uncertainty_low(u):
    return torch.mean((u < 0.25).float(), dim=-1) > 1/3

def _is_pose_consistent_with_box(p2d, box):
    pmin, pmax = torch.min(p2d, dim=-2).values, torch.max(p2d, dim=-2).values
    bmin, bmax = box[..., :2], box[..., :2] + box[..., 2:4]
    inter = torch.prod(torch.relu(torch.minimum(pmax, bmax) - torch.maximum(pmin, bmin)), dim=-1)
    return inter > 0.25 * torch.prod(box[..., 2:4], dim=-1)

def _pose_nms(poses, scores, valid):
    idx = torch.squeeze(torch.argwhere(valid), 1)
    if len(idx) == 0:
        return idx
    pp, ss = poses[idx], scores[idx]
    sq = torch.mean(pp**2, dim=(-2,-1), keepdim=True)
    sq1, sq2 = sq.unsqueeze(0), sq.unsqueeze(1)
    msq = (sq1 + sq2) / 2
    dists = torch.linalg.norm(torch.sqrt(msq/sq1)*pp.unsqueeze(0) - torch.sqrt(msq/sq2)*pp.unsqueeze(1), dim=-1)
    sim = torch.mean(torch.relu(1 - torch.topk(dists, k=max(1, pp.shape[-2]//5), sorted=False).values / 300), dim=-1)
    order = torch.argsort(ss, descending=True)
    keep, suppressed = [], torch.zeros(len(idx), dtype=torch.bool, device=poses.device)
    for i in range(len(idx)):
        j = order[i]
        if not suppressed[j]:
            keep.append(j)
            suppressed |= sim[j] > 0.4
    return idx[torch.stack(keep)] if keep else idx[:0]

def _scale_align(poses):
    sq = torch.mean(poses**2, dim=(-2,-1), keepdim=True)
    return poses * torch.sqrt(torch.mean(sq, dim=-3, keepdim=True) / sq)

def _weighted_mean(x, w, dim=-2, keepdim=False):
    return (x * w).sum(dim=dim, keepdim=keepdim) / w.sum(dim=dim, keepdim=keepdim)

def _weighted_geometric_median(x, w, n_iter=10, dim=-2, eps=1e-1, keepdim=False):
    if dim < 0:
        dim = x.ndim + dim
    if w is None:
        w = torch.ones_like(x[..., :1])
    else:
        w = w.unsqueeze(-1)
    new_w = w
    y = _weighted_mean(x, new_w, dim=dim, keepdim=True)
    for _ in range(n_iter):
        new_w = w / (torch.norm(x - y, dim=-1, keepdim=True) + eps)
        y = _weighted_mean(x, new_w, dim=dim, keepdim=True)
    return (y if keepdim else y.squeeze(dim)), new_w.squeeze(-1)

# Multi-person pipeline

class MultipersonNLF:
    """Multi-person NLF inference pipeline (non-parametric only)."""

    def __init__(self, crop_model, model_patcher=None, detector=None, canonical_points=None, num_vertices=1024, pad_white_pixels=True):
        self.crop_model = crop_model
        self.model_patcher = model_patcher
        self.detector = detector
        self.pad_white_pixels = pad_white_pixels
        self.canonical_points = canonical_points
        self.num_vertices = num_vertices
        self._weights = None
        self._tta_cache = {}
        self._intrinsic_cache = {}

    def _get_weights(self, device):
        if self._weights is None or self._weights['w_tensor'].device != device:
            self._weights = self.crop_model.get_weights_for_canonical_points(self.canonical_points.to(device))
        return self._weights

    def detect_and_estimate(self, images, intrinsic_matrix=None, default_fov_degrees=55.0,
                            internal_batch_size=64, num_aug=1, rot_aug_max_degrees=25.0,
                            detector_threshold=0.3, detector_nms_iou_threshold=0.7,
                            max_detections=150, suppress_implausible_poses=True, boxes=None):
        device = images.device

        if boxes is None:
            assert self.detector is not None, "No detector and no pre-computed boxes provided"
            boxes = self.detector.detect(images, threshold=detector_threshold, nms_iou_threshold=detector_nms_iou_threshold, max_detections=max_detections)

        # Gamma decode to linear light after detection (detector needs sRGB input)
        images = images.to(dtype=torch.float16).pow(2.2)
        weights = self._get_weights(device)
        result = self._estimate_poses_batched(
            images, boxes, weights, intrinsic_matrix=intrinsic_matrix,
            default_fov_degrees=default_fov_degrees, internal_batch_size=internal_batch_size,
            num_aug=num_aug, rot_aug_max_degrees=rot_aug_max_degrees,
            suppress_implausible_poses=suppress_implausible_poses,
        )

        # Split: first num_vertices are mesh vertices, rest are joints
        nv = self.num_vertices
        result['poses3d'] = [
            (p[:, nv:, :] if p.shape[0] > 0 else p[:, :0, :])
            for p in result['poses3d']
        ]
        return result

    def _estimate_poses_batched(self, images, boxes, weights, intrinsic_matrix=None,
                                default_fov_degrees=55.0, internal_batch_size=64,
                                antialias_factor=1, num_aug=1, rot_aug_max_degrees=25.0,
                                suppress_implausible_poses=True):
        if sum(len(b) for b in boxes) == 0:
            return self._predict_empty(images, weights)

        n_images, device = len(images), images.device

        if intrinsic_matrix is None:
            cache_key = (default_fov_degrees, images.shape[2], images.shape[3], device)
            if cache_key not in self._intrinsic_cache:
                self._intrinsic_cache[cache_key] = intrinsic_matrix_from_field_of_view(default_fov_degrees, list(images.shape[2:4]), device=device)
            intrinsic_matrix = self._intrinsic_cache[cache_key]
        if len(intrinsic_matrix) == 1:
            intrinsic_matrix = intrinsic_matrix.repeat(n_images, 1, 1)

        n_box = torch.tensor([len(b) for b in boxes], device=device)
        n_box_list = [len(b) for b in boxes]
        intrinsic_matrix = torch.repeat_interleave(intrinsic_matrix, n_box, dim=0)

        camspace_up = torch.tensor([[0, -1, 0]], device=device, dtype=torch.float32).expand(intrinsic_matrix.shape[0], -1)

        # TTA params (cached — same for every frame with same num_aug)
        tta_key = (num_aug, rot_aug_max_degrees, device)
        if tta_key not in self._tta_cache:
            aug_gammas = _linspace(0.6, 1.0, num_aug, dtype=torch.float32, device=device)
            aug_angles = _linspace(-rot_aug_max_degrees * torch.pi / 180, rot_aug_max_degrees * torch.pi / 180, num_aug, dtype=torch.float32, device=device)
            aug_scales = (torch.tensor([1.0], device=device) if num_aug == 1 else
                          torch.cat([_linspace(0.8, 1.0, num_aug//2, endpoint=False, dtype=torch.float32, device=device),
                                     torch.linspace(1.0, 1.1, num_aug - num_aug//2, device=device)]))
            aug_flip = (torch.arange(num_aug, device=device) - num_aug // 2) % 2 != 0
            flipmat = torch.tensor([[-1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32, device=device)
            aug_rfmat = torch.where(aug_flip[:, None, None], flipmat, torch.eye(3, device=device)) @ rotation_mat(-aug_angles, 'z')
            self._tta_cache[tta_key] = (aug_gammas, aug_scales, aug_flip, aug_rfmat)
        aug_gammas, aug_scales, aug_flip, aug_rfmat = self._tta_cache[tta_key]

        poses3d_flat, uncert_flat = self._predict_in_batches(
            images, weights, intrinsic_matrix, camspace_up,
            boxes, internal_batch_size, aug_flip, aug_rfmat, aug_gammas, aug_scales, antialias_factor)

        poses3d_flat = _scale_align(poses3d_flat)
        mean = poses3d_flat.mean(dim=(-3, -2), keepdim=True)
        sub, final_w = _weighted_geometric_median(
            (poses3d_flat - mean), uncert_flat ** -1.5, dim=-3, n_iter=10, eps=50.0)
        poses3d_flat = sub + mean.squeeze(1)
        uncert_flat = _weighted_mean(uncert_flat, final_w, dim=-2)

        # Project 3D -> 2D (no distortion)
        projected = poses3d_flat / torch.clamp(poses3d_flat[..., 2:], min=0.1)
        poses2d_flat = torch.einsum('bnk,bjk->bnj', projected, intrinsic_matrix[:, :2, :])

        poses3d = list(torch.split(poses3d_flat, n_box_list))
        poses2d = list(torch.split(poses2d_flat, n_box_list))
        uncert = list(torch.split(uncert_flat, n_box_list))

        if suppress_implausible_poses:
            new_boxes, new_p3, new_p2, new_u = [], [], [], []
            for b, p3, p2, u in zip(boxes, poses3d, poses2d, uncert):
                ok = torch.logical_and(_is_uncertainty_low(u), _is_pose_consistent_with_box(p2, b))
                idx = _pose_nms(p3, b[..., 4] / u.mean(dim=-1), ok)
                new_boxes.append(b[idx])
                new_p3.append(p3[idx])
                new_p2.append(p2[idx])
                new_u.append(u[idx])
            boxes, poses3d, poses2d, uncert = new_boxes, new_p3, new_p2, new_u

        n_box_list = [len(b) for b in boxes]
        if sum(n_box_list) == 0:
            return self._predict_empty(images, weights)

        return dict(boxes=boxes, poses3d=poses3d, poses2d=poses2d, uncertainties=uncert)

    def _predict_in_batches(self, images, weights, intr, up, boxes, batch_size, aug_flip, aug_rf, aug_g, aug_s, aa):
        num_aug = len(aug_g)
        bpb = batch_size // num_aug
        boxes_flat = torch.cat(boxes, dim=0)
        img_ids = torch.repeat_interleave(torch.arange(len(boxes), device=boxes_flat.device), torch.tensor([len(b) for b in boxes], device=boxes_flat.device))

        if bpb == 0:
            return self._predict_single_batch(images, weights, intr, up, boxes_flat, img_ids, aug_rf, aug_flip, aug_s, aug_g, aa)

        batches_p, batches_u = [], []
        for i in range(math.ceil(len(boxes_flat) / bpb)):
            s = slice(i * bpb, (i+1) * bpb)
            p, u = self._predict_single_batch(images, weights, intr[s], up[s], boxes_flat[s], img_ids[s], aug_rf, aug_flip, aug_s, aug_g, aa)
            batches_p.append(p)
            batches_u.append(u)
        return torch.cat(batches_p), torch.cat(batches_u)

    def _predict_single_batch(self, images, weights, intr, up, boxes, img_ids, aug_rf, aug_flip, aug_s, aug_g, aa):
        crops, new_intr, R = self._get_crops(images, intr, up, boxes, img_ids, aug_rf, aug_s, aug_g, aa)
        res = self.crop_model.input_resolution
        crops_flat = crops.reshape(-1, 3, res, res)
        new_intr_flat = new_intr.reshape(-1, 3, 3)
        flip_flat = aug_flip.repeat_interleave(crops.shape[1])
        poses_flat, uncert_flat = self.crop_model.predict_multi_same_weights(crops_flat, new_intr_flat, weights, flip_flat)
        n_cases, n_joints = crops.shape[1], poses_flat.shape[-2]
        poses = poses_flat.reshape(-1, n_cases, n_joints, 3) @ R
        return poses.transpose(0, 1), uncert_flat.reshape(-1, n_cases, n_joints).transpose(0, 1)

    def _get_crops(self, images, intr, up, boxes, img_ids, aug_rf, aug_s, aug_g, aa):
        R_noaug, box_scales = self._get_rotation_and_scale(intr, up, boxes)
        device, num_box, num_aug, res = images.device, boxes.shape[0], aug_g.shape[0], self.crop_model.input_resolution
        crop_scales = aug_s[:, None] * box_scales[None, :]

        new_intr = torch.cat([
            torch.cat([intr[None, :, :2, :2] * crop_scales[:, :, None, None],
                        torch.full((num_aug, num_box, 2, 1), res/2, dtype=torch.float32, device=device)], dim=3),
            torch.cat([torch.zeros((num_aug, num_box, 1, 2), dtype=torch.float32, device=device),
                        torch.ones((num_aug, num_box, 1, 1), dtype=torch.float32, device=device)], dim=3),
        ], dim=2)

        R = aug_rf[:, None] @ R_noaug
        new_inv = torch.linalg.inv(new_intr @ R)
        if aa > 1:
            new_inv = new_inv @ _corner_aligned_scale_mat(1/aa, device=device)

        if self.pad_white_pixels:
            images = images.neg().add(1)

        crops = _warp_images_pyramid(images,
            intrinsic_matrix=intr.tile([num_aug, 1, 1]), new_invprojmats=new_inv.reshape(-1, 3, 3),
            crop_scales=crop_scales.reshape(-1) * aa,
            output_shape=(res*aa, res*aa), image_ids=img_ids.tile([num_aug]))

        if self.pad_white_pixels:
            crops = crops.neg().add(1).clamp(0, 1)
        if aa == 2:
            crops = F.avg_pool2d(crops, 2, 2)
        elif aa == 4:
            crops = F.avg_pool2d(crops, 4, 4)

        crops = crops.reshape(num_aug, num_box, 3, res, res)
        crops **= (aug_g.to(crops.dtype) / 2.2).reshape(-1, 1, 1, 1, 1)
        return crops, new_intr, R

    def _get_rotation_and_scale(self, intr, up, boxes):
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        pts = to_homogeneous(torch.stack([
            torch.stack([x+w/2, y+h/2], dim=1), torch.stack([x+w/2, y], dim=1),
            torch.stack([x+w, y+h/2], dim=1), torch.stack([x+w/2, y+h], dim=1),
            torch.stack([x, y+h/2], dim=1),
        ], dim=1))
        # No distortion: just apply inverse intrinsics directly
        cam = torch.einsum('bpc,bCc->bpC', pts, torch.linalg.inv(intr))
        cam = to_homogeneous(cam[..., :2])
        R = lookat_matrix(cam[:, 0], up)
        side = _project(torch.einsum('bpc,bCc->bpC', cam[:, 1:5], intr @ R))
        box_size = torch.maximum(
            torch.linalg.norm(side[:, 0] - side[:, 2], dim=-1),
            torch.linalg.norm(side[:, 1] - side[:, 3], dim=-1))
        return R, torch.tensor(self.crop_model.input_resolution, dtype=box_size.dtype, device=box_size.device) / box_size

    def _predict_empty(self, images, weights):
        device, n = images.device, images.shape[0]
        nj = weights['w_tensor'].shape[0]
        empty = torch.zeros((0, 5), dtype=torch.float32, device=device)
        return dict(
            boxes=[empty]*n, poses3d=[torch.zeros((0, nj, 3), dtype=torch.float32, device=device)]*n,
            poses2d=[torch.zeros((0, nj, 2), dtype=torch.float32, device=device)]*n,
            uncertainties=[torch.zeros((0, nj), dtype=torch.float32, device=device)]*n)
