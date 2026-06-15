# NLF model: config, tensor utilities, 3D reconstruction, weight field, localizer head, and NLFModel

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.ops
import comfy.model_management


# Tensor utilities

def mean_stdev_masked(input_tensor, is_valid, items_dim, dimensions_dim, fixed_ref=None):
    if fixed_ref is not None:
        mean = fixed_ref
    else:
        mean = reduce_mean_masked(input_tensor, is_valid, dim=[items_dim], keepdim=True)
    centered = input_tensor - mean
    n_new_dims = input_tensor.ndim - is_valid.ndim
    is_valid = is_valid.reshape(is_valid.shape + (1,) * n_new_dims)
    n_valid = is_valid.sum(dim=items_dim, keepdim=True, dtype=input_tensor.dtype)
    sum_sq = reduce_sum_masked(torch.square(centered), is_valid, dim=[items_dim, dimensions_dim], keepdim=True)
    return mean, torch.sqrt(torch.nan_to_num(sum_sq / n_valid) + 1e-10)


def reduce_mean_masked(input_tensor, is_valid, dim=None, keepdim=False):
    d = [] if dim is None else dim
    if is_valid is None:
        return torch.mean(input_tensor, dim=d, keepdim=keepdim)
    if dim is None and not keepdim:
        return torch.masked_select(input_tensor, is_valid).mean()
    n_new_dims = input_tensor.ndim - is_valid.ndim
    is_valid = is_valid.reshape(is_valid.shape + (1,) * n_new_dims)
    replaced = torch.where(is_valid, input_tensor, torch.zeros_like(input_tensor))
    return torch.nan_to_num(torch.sum(replaced, dim=d, keepdim=keepdim) / torch.sum(is_valid, dim=d, keepdim=keepdim, dtype=input_tensor.dtype))


def reduce_sum_masked(input_tensor, is_valid, dim=None, keepdim=False):
    if dim is None and not keepdim:
        return torch.masked_select(input_tensor, is_valid).sum()
    n_new_dims = input_tensor.ndim - is_valid.ndim
    is_valid = is_valid.reshape(is_valid.shape + (1,) * n_new_dims)
    replaced = torch.where(is_valid, input_tensor, torch.zeros_like(input_tensor))
    return torch.sum(replaced, dim=([] if dim is None else dim), keepdim=keepdim)


def softmax_nd(target, dim=[-1]):
    dim = [d if d >= 0 else d + target.ndim for d in dim]
    assert sorted(dim) == list(range(min(dim), max(dim) + 1))
    flattened = target.flatten(start_dim=min(dim), end_dim=max(dim))
    return torch.softmax(flattened, dim=min(dim)).reshape(target.shape)


def soft_argmax(inp, dim=[-1]):
    return decode_heatmap(softmax_nd(inp, dim=dim), dim=dim)


def decode_heatmap(inp, dim=[-1], output_coord_dim=-1):
    result = []
    dim = [d if d >= 0 else d + inp.ndim for d in dim]
    for d in dim:
        other = [x for x in dim if x != d]
        summed = torch.sum(inp, dim=other, keepdim=True)
        coords = _linspace(0.0, 1.0, inp.shape[d], dtype=inp.dtype, device=summed.device)
        decoded = torch.unsqueeze(torch.tensordot(summed, coords, dims=([d], [0])), d)
        for hd in sorted(dim)[::-1]:
            decoded = decoded.squeeze(hd)
        result.append(decoded)
    return torch.stack(result, dim=output_coord_dim)


def _linspace(start, stop, num, dtype=None, device=None, endpoint=True):
    if endpoint and num == 1:
        s = torch.as_tensor(start, device=device, dtype=dtype)
        e = torch.as_tensor(stop, device=device, dtype=dtype)
        return torch.mean(torch.stack([s, e], dim=0), dim=0, keepdim=True)
    if not endpoint and num > 1:
        step = (stop - start) / num
        return torch.linspace(start, stop - step, num, device=device, dtype=dtype)
    return torch.linspace(start, stop, num, device=device, dtype=dtype)


def dynamic_partition(x, partitions, num_partitions):
    return [x[partitions == i] for i in range(num_partitions)]


def dynamic_stitch(index_lists, value_lists):
    indices = torch.cat(index_lists, dim=0)
    values = torch.cat(value_lists, dim=0)
    return values[torch.argsort(indices)]


# 3D reconstruction

def to_homogeneous(x):
    return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)


def reconstruct_absolute(
    coords2d, coords3d_rel, intrinsics, proc_side, stride, centered_stride,
    mix_3d_inside_fov=None, weak_perspective=False, point_validity_mask=None,
    border_factor1=0.75, border_factor2=None, mix_based_on_3d=True,
):
    inv_intrinsics = torch.linalg.inv(intrinsics.to(coords2d.dtype))
    coords2d_norm = (to_homogeneous(coords2d) @ inv_intrinsics.transpose(1, 2))[..., :2]
    if border_factor2 is None:
        border_factor2 = border_factor1

    in_fov1 = _is_within_fov(coords2d, proc_side, stride, centered_stride, border_factor1)
    if point_validity_mask is not None:
        in_fov1 = torch.logical_and(in_fov1, point_validity_mask)

    if weak_perspective:
        ref = _reconstruct_ref_weakpersp(coords2d_norm, coords3d_rel, in_fov1)
    else:
        ref = _reconstruct_ref_fullpersp(coords2d_norm, coords3d_rel, in_fov1)

    abs_3d = coords3d_rel + ref.unsqueeze(1)
    abs_2d = to_homogeneous(coords2d_norm) * (coords3d_rel[..., 2] + ref[:, 2].unsqueeze(-1)).unsqueeze(-1)

    if mix_3d_inside_fov is not None:
        abs_2d = torch.lerp(abs_2d, abs_3d, mix_3d_inside_fov)

    proj = _project_pose(abs_3d if mix_based_on_3d else abs_2d, intrinsics)
    in_fov2 = torch.logical_and(
        _is_within_fov(proj, proc_side, stride, centered_stride, border_factor2),
        abs_3d[..., 2] > 0.001,
    )
    return torch.where(in_fov2[..., np.newaxis], abs_2d, abs_3d)


def _reconstruct_ref_weakpersp(norm2d, rel3d, mask):
    _, stdev3d = mean_stdev_masked(rel3d[..., :2], mask, items_dim=1, dimensions_dim=2)
    mean2d, stdev2d = mean_stdev_masked(norm2d[..., :2], mask, items_dim=1, dimensions_dim=2)
    stdev2d = torch.maximum(stdev2d, torch.tensor(1e-5))
    stdev3d = torch.maximum(stdev3d, torch.tensor(1e-5))
    old_mean = reduce_mean_masked(rel3d, mask, dim=[1], keepdim=True)
    new_mean = to_homogeneous(mean2d) * torch.nan_to_num(stdev3d / stdev2d)
    return torch.squeeze(new_mean - old_mean, 1)


def _reconstruct_ref_fullpersp(norm2d, rel3d, mask):
    n_batch, n_points = norm2d.shape[0], norm2d.shape[1]
    eyes2 = torch.eye(2, device=norm2d.device, dtype=norm2d.dtype).unsqueeze(0).repeat(n_batch, n_points, 1)
    scale2d, reshaped2d = _rms_normalize_reshape(norm2d, mask, n_points)
    A = torch.cat([eyes2, -reshaped2d], dim=2)
    rel_backproj = norm2d * rel3d[:, :, 2:] - rel3d[:, :, :2]
    scale_rb, b = _rms_normalize_reshape(rel_backproj, mask, n_points)
    weights = (mask.to(norm2d.dtype) + 1e-8).repeat_interleave(2, 1)
    ref = _lstsq_cholesky(A, b, weights, l2_regularizer=1e-4)
    ref = torch.cat([ref[:, :2] * scale_rb, ref[:, 2:] * (scale_rb / scale2d)], dim=1)
    return torch.squeeze(ref, dim=-1)


def _rms_normalize_reshape(x, mask, n_points):
    scale = torch.sqrt(reduce_mean_masked(torch.square(x), mask, dim=[1, 2], keepdim=True) + 1e-10)
    return scale, (x / scale).reshape(-1, n_points * 2, 1)


def _lstsq_cholesky(matrix, rhs, weights, l2_regularizer=None):
    wm = weights.unsqueeze(-1) * matrix
    gram = wm.mT @ matrix
    if l2_regularizer is not None:
        gram.diagonal(dim1=-2, dim2=-1).add_(l2_regularizer)
    chol, _ = torch.linalg.cholesky_ex(gram)
    return torch.cholesky_solve(wm.mT @ rhs, chol)


def _is_within_fov(imcoords, proc_side, stride, centered_stride, border_factor=0.75):
    offset = 0.0 if centered_stride else -stride / 2.0
    lower = stride * border_factor + offset
    upper = proc_side - stride * border_factor + offset
    return torch.all(torch.logical_and(imcoords >= lower, imcoords <= upper), dim=-1)


def _project_pose(coords3d, intrinsic_matrix):
    projected = coords3d / torch.clamp(coords3d[..., 2:], min=0.1)
    return torch.einsum('bnk,bjk->bnj', projected, intrinsic_matrix[..., :2, :])


def intrinsic_matrix_from_field_of_view(fov_degrees, imshape, device=None):
    imshape_t = torch.tensor(imshape, dtype=torch.float32, device=device)
    fov_rad = fov_degrees * (torch.pi / 180)
    f = torch.max(imshape_t) / (torch.tan(torch.tensor(fov_rad / 2, device=device)) * 2)
    _0, _1 = torch.tensor(0, dtype=torch.float32, device=device), torch.tensor(1, dtype=torch.float32, device=device)
    return torch.stack([f, _0, (imshape_t[1]-1)/2, _0, f, (imshape_t[0]-1)/2, _0, _0, _1], dim=-1).unflatten(-1, (3, 3)).unsqueeze(0)


def lookat_matrix(forward_vector, up_vector):
    new_z = F.normalize(forward_vector, dim=-1)
    new_x = torch.linalg.cross(new_z, up_vector)
    new_x_alt = torch.stack([new_z[:, 2], torch.zeros_like(new_z[:, 2]), -new_z[:, 0]], dim=1)
    new_x = torch.where(torch.linalg.norm(new_x, dim=-1, keepdim=True) == 0, new_x_alt, new_x)
    new_x = F.normalize(new_x, dim=-1)
    return torch.stack([new_x, torch.linalg.cross(new_z, new_x), new_z], dim=1)


def rotation_mat(angle, rot_axis):
    sin, cos = torch.sin(angle), torch.cos(angle)
    _0, _1 = torch.zeros_like(angle), torch.ones_like(angle)
    if rot_axis == 'x':
        elems = [_1, _0, _0, _0, cos, sin, _0, -sin, cos]
    elif rot_axis == 'y':
        elems = [cos, _0, -sin, _0, _1, _0, sin, _0, cos]
    else:
        elems = [cos, -sin, _0, sin, cos, _0, _0, _0, _1]
    return torch.stack(elems, dim=-1).unflatten(-1, (3, 3))


# Coordinate conversion

def heatmap_to_image(coords, proc_side, stride, centered_stride):
    last_pix = proc_side - 1
    out = coords * (last_pix - (last_pix % stride))
    if centered_stride:
        out = out + stride // 2
    return out


def heatmap_to_metric(coords, proc_side, stride, centered_stride, box_size_m):
    xy = heatmap_to_image(coords[..., :2], proc_side, stride, centered_stride) * box_size_m / proc_side
    return torch.cat([xy, coords[..., 2:] * box_size_m], dim=-1)


# Weight field

class LearnableFourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None, operations=None):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.linear = operations.Linear(in_features, out_features // 2, bias=False, device=device, dtype=dtype)

    def forward(self, inp):
        x = self.linear(inp)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class GPSNet(nn.Module):
    def __init__(self, pos_enc_dim=512, hidden_dim=2048, output_dim=1024, device=None, dtype=None, operations=None):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.factor = 1 / np.sqrt(np.float32(pos_enc_dim))
        self.register_buffer('mini', torch.zeros(3), persistent=False)
        self.register_buffer('maxi', torch.ones(3), persistent=False)
        self.register_buffer('center', torch.zeros(3), persistent=False)
        self.learnable_fourier = LearnableFourierFeatures(3, pos_enc_dim, device=device, dtype=dtype, operations=operations)
        self.mlp = nn.Sequential(
            operations.Linear(pos_enc_dim, hidden_dim, device=device, dtype=dtype), nn.GELU(),
            operations.Linear(hidden_dim, output_dim, device=device, dtype=dtype),
        )

    def forward(self, inp):
        x = self.learnable_fourier((inp - self.center) / (self.maxi - self.mini)) * self.factor
        return self.mlp(x)


class GPSField(nn.Module):
    def __init__(self, posenc_dim=1024, backbone_link_dim=512, depth=8,
                 hidden_size=384, hidden_layers=1, device=None, dtype=None, operations=None):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.posenc_dim = posenc_dim
        out_dim = (backbone_link_dim + 1) * (depth + 2)
        layer_dims = [hidden_size] * hidden_layers + [out_dim]
        self.gps_net = GPSNet(pos_enc_dim=512, hidden_dim=2048, output_dim=posenc_dim, device=device, dtype=dtype, operations=operations)
        self.pred_mlp = nn.Sequential()
        self.pred_mlp.append(operations.Linear(posenc_dim, layer_dims[0], device=device, dtype=dtype))
        self.pred_mlp.append(nn.GELU())
        for i in range(1, len(layer_dims) - 1):
            self.pred_mlp.append(operations.Linear(layer_dims[i-1], layer_dims[i], device=device, dtype=dtype))
            self.pred_mlp.append(nn.GELU())
        self.pred_mlp.append(operations.Linear(layer_dims[-2], layer_dims[-1], device=device, dtype=dtype))
        self.register_buffer('r_sqrt_eigva', torch.ones(posenc_dim), persistent=False)

    def forward(self, inp):
        lbo = self.gps_net(inp.reshape(-1, 3))[..., :self.posenc_dim]
        lbo = lbo.reshape(inp.shape[:-1] + (self.posenc_dim,)) * self.r_sqrt_eigva[:self.posenc_dim] * 0.1
        return self.pred_mlp(lbo)


# Localizer head

class LocalizerHead(nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        if operations is None:
            operations = comfy.ops.disable_weight_init
        self.uncert_bias = 0.0
        self.uncert_bias2 = 0.001
        self.depth = 8
        self.stride_test = 32
        self.centered_stride = True
        self.box_size_m = 2.2
        self.proc_side = 384
        self.backbone_link_dim = 512
        self.fix_uncert_factor = False
        self.mix_3d_inside_fov = 0.5
        self.weak_perspective = False
        self.weight_field = GPSField(device=device, dtype=dtype, operations=operations)
        self.layer = nn.Sequential(
            operations.Conv2d(1280, 512, kernel_size=1, bias=False, device=device, dtype=dtype),
            nn.BatchNorm2d(512, eps=1e-3),
            nn.SiLU(),
        )

    def predict_same_canonicals(self, features, canonical_positions):
        weights = self.weight_field(canonical_positions)
        features_processed = self.layer(features)
        coords2d, coords3d, uncertainties = self._apply_weights_same_canonicals(features_processed, weights)
        return self._to_image_metric(coords2d, coords3d, uncertainties)

    def _apply_weights_same_canonicals(self, features, weights):
        w, b = self._transpose_weights(weights.to(features.dtype), features.shape[1])
        return self._apply_weights_impl(features, w, b)

    def _apply_weights_impl(self, features, w_tensor, b_tensor):
        n_out = 2 + self.depth
        dtype = features.dtype
        w_flat = w_tensor.flatten(0, 1).unsqueeze(-1).unsqueeze(-1).to(dtype)
        b_flat = b_tensor.reshape(-1).to(dtype)
        logits = comfy.model_management.cast_to(F.conv2d(features, w_flat, bias=b_flat), dtype=torch.float32)
        logits = logits.unflatten(1, (-1, n_out))
        uncertainty_map = logits[:, :, 0]
        coords_xy = soft_argmax(logits[:, :, 1], dim=[3, 2])
        heatmap25d = softmax_nd(logits[:, :, 2:], dim=[4, 3, 2])
        heatmap2d = heatmap25d.sum(dim=2)
        uncertainties = torch.einsum('nphw,nphw->np', uncertainty_map, heatmap2d.detach())
        uncertainties = F.softplus(uncertainties + self.uncert_bias) + self.uncert_bias2
        coords25d = decode_heatmap(heatmap25d, dim=[4, 3, 2])
        return coords25d[..., :2], torch.cat([coords_xy, coords25d[..., 2:]], dim=-1), uncertainties

    def _transpose_weights(self, weights, n_in_channels):
        n_out = 2 + self.depth
        resh = weights.unflatten(-1, (n_in_channels + 1, n_out))
        return resh[..., :-1, :].permute(0, 2, 1).contiguous(), resh[..., -1, :].contiguous()

    def _to_image_metric(self, coords2d, coords3d, uncertainties):
        return (
            heatmap_to_image(coords2d, self.proc_side, self.stride_test, self.centered_stride),
            heatmap_to_metric(coords3d, self.proc_side, self.stride_test, self.centered_stride, self.box_size_m),
            uncertainties,
        )

    def get_weights_for_canonical_points(self, canonical_points):
        # Run normal + x-flipped canonical points through the field in one batch
        flipped = canonical_points * torch.tensor([-1, 1, 1], dtype=canonical_points.dtype, device=canonical_points.device)
        both = torch.cat([canonical_points, flipped], dim=0)
        both_weights = self.weight_field(both)
        weights, weights_fl = both_weights.chunk(2, dim=0)
        # Cast to match the conv layer dtype (typically fp16)
        conv_dtype = self.layer[0].weight.dtype
        w, b = self._transpose_weights(weights.to(conv_dtype), self.backbone_link_dim)
        wf, bf = self._transpose_weights(weights_fl.to(conv_dtype), self.backbone_link_dim)
        return dict(w_tensor=w, b_tensor=b, w_tensor_flipped=wf, b_tensor_flipped=bf)

    def decode_features_multi_same_weights(self, features, weights, flip_canonicals_per_image):
        flip_ind = flip_canonicals_per_image.to(torch.int32)
        nfl, fl = dynamic_partition(features, flip_ind, 2)
        idx = dynamic_partition(torch.arange(features.shape[0], device=flip_ind.device), flip_ind, 2)
        nfl_c2, nfl_c3, nfl_u = self._apply_weights_impl(nfl, weights['w_tensor'], weights['b_tensor'])
        fl_c2, fl_c3, fl_u = self._apply_weights_impl(fl, weights['w_tensor_flipped'], weights['b_tensor_flipped'])
        c2 = dynamic_stitch(idx, [nfl_c2, fl_c2])
        c3 = dynamic_stitch(idx, [nfl_c3, fl_c3])
        u = dynamic_stitch(idx, [nfl_u, fl_u])
        return self._to_image_metric(c2, c3, u)

    def reconstruct_absolute_coords(self, coords2d, coords3d, uncertainties, intrinsic_matrix):
        coords3d_abs = reconstruct_absolute(
            coords2d, coords3d, intrinsic_matrix,
            proc_side=self.proc_side, stride=self.stride_test, centered_stride=self.centered_stride,
            weak_perspective=self.weak_perspective, mix_3d_inside_fov=0.5,
            point_validity_mask=uncertainties < 0.3, border_factor1=1.0, border_factor2=0.6, mix_based_on_3d=True,
        ) * 1000
        factor = 1 if self.fix_uncert_factor else 3
        return coords3d_abs, uncertainties * factor


# NLFModel

class NLFModel(nn.Module):
    def __init__(self, n_left_joints, n_center_joints, n_joints, device=None, dtype=None, operations=None):
        super().__init__()
        from .backbone import build_backbone
        self.backbone, _ = build_backbone(device=device, dtype=dtype, operations=operations)
        self.heatmap_head = LocalizerHead(device=device, dtype=dtype, operations=operations)
        self.input_resolution = 384
        self.register_buffer('inv_permutation', torch.zeros(n_joints, dtype=torch.int32), persistent=False)
        self.register_buffer('canonical_locs_init', torch.zeros((n_joints, 3), dtype=torch.float32), persistent=False)
        self.register_buffer('canonical_delta_mask', torch.zeros(n_joints, dtype=torch.float32), persistent=False)
        self.canonical_lefts = nn.Parameter(torch.zeros((n_left_joints, 3), dtype=torch.float32))
        self.canonical_centers = nn.Parameter(torch.zeros((n_center_joints, 2), dtype=torch.float32))

    @classmethod
    def from_state_dict(cls, sd, device=None, dtype=None, operations=None):
        model = cls(sd['canonical_lefts'].shape[0], sd['canonical_centers'].shape[0],
                     sd['canonical_locs_init'].shape[0], device, dtype, operations)
        _assign_non_persistent_buffers(model, sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        unexpected = [k for k in unexpected if not _is_non_persistent_buffer(model, k)]
        if missing:
            logging.warning(f"NLFModel: missing keys: {missing}")
        if unexpected:
            logging.warning(f"NLFModel: unexpected keys: {unexpected}")
        return model

    def canonical_locs(self):
        rights = torch.cat([-self.canonical_lefts[:, :1], self.canonical_lefts[:, 1:]], dim=1)
        centers = torch.cat([torch.zeros_like(self.canonical_centers[:, :1]), self.canonical_centers], dim=1)
        permuted = torch.cat([self.canonical_lefts, rights, centers], dim=0)
        return permuted.index_select(0, self.inv_permutation) * self.canonical_delta_mask[:, None] + self.canonical_locs_init

    def _to_float32(self, *tensors):
        return tuple(comfy.model_management.cast_to(t, dtype=torch.float32) for t in tensors)

    def predict_multi_same_canonicals(self, image, intrinsic_matrix, canonical_points):
        features = self.backbone(image)
        c2, c3, u = self.heatmap_head.predict_same_canonicals(features, canonical_points)
        with torch.amp.autocast('cuda', enabled=False):
            c2, c3, u, intrinsic_matrix = self._to_float32(c2, c3, u, intrinsic_matrix)
            return self.heatmap_head.reconstruct_absolute_coords(c2, c3, u, intrinsic_matrix)

    def get_features(self, image):
        return self.heatmap_head.layer(self.backbone(image))

    def predict_multi_same_weights(self, image, intrinsic_matrix, weights, flip_canonicals_per_image):
        features = self.get_features(image)
        c2, c3, u = self.heatmap_head.decode_features_multi_same_weights(features, weights, flip_canonicals_per_image)
        with torch.amp.autocast('cuda', enabled=False):
            c2, c3, u, intrinsic_matrix = self._to_float32(c2, c3, u, intrinsic_matrix)
            return self.heatmap_head.reconstruct_absolute_coords(c2, c3, u, intrinsic_matrix)

    def get_weights_for_canonical_points(self, canonical_points):
        return self.heatmap_head.get_weights_for_canonical_points(canonical_points)


def _assign_non_persistent_buffers(module, sd, prefix=''):
    non_persistent = getattr(module, '_non_persistent_buffers_set', set())
    for name in module._buffers:
        key = f"{prefix}{name}" if prefix else name
        if key in sd and name in non_persistent:
            module._buffers[name] = sd[key]
    for name, child in module._modules.items():
        if child is not None:
            _assign_non_persistent_buffers(child, sd, f"{prefix}{name}." if prefix else f"{name}.")


def _is_non_persistent_buffer(module, key):
    parts = key.split('.')
    current = module
    for part in parts[:-1]:
        current = getattr(current, part, None)
        if current is None:
            return False
    return parts[-1] in getattr(current, '_non_persistent_buffers_set', set()) or parts[-1] in current._buffers
