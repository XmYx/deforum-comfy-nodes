"""Forward-splatting 3D warp that reports where content is new.

deforum warps by building a per-pixel offset field from the camera projection
and then calling ``grid_sample`` -- a *backward* warp. Backward sampling always
produces a dense image, which sounds good until you notice what it does at a
disocclusion: there is no source pixel for newly revealed area, so it stretches
whatever sat at the edge. That is the smeared "taffy" look.

Splatting *forward* with a z-buffer also fixes the other half of the problem,
because when several source pixels land on one destination the nearest surface
wins instead of whatever happened to be sampled last.

Forward warping leaves holes where backward warping does not. Those holes are
the point: they mark exactly the pixels the sampler should be allowed to
invent. Returned as a mask, they can drive a per-pixel denoise strength, which
is the thing a single ``strength_schedule`` value cannot express.

The camera maths deliberately mirrors ``anim_frame_warp_3d`` so this warp moves
the frame the same way the existing one does, using deforum's own bundled
``py3d_tools`` rather than adding a pytorch3d dependency.
"""

import math

import numpy as np
import torch

TRANSLATION_SCALE = 1.0 / 200.0  # matches deforum, which matches Disco


def camera_offset(depth, anim_args, keys, frame_idx, width, height, device):
    """The per-pixel offset field deforum's 3D warp would apply, in [-1, 1].

    Returns ``None`` when the projection cannot be built, so callers can fall
    back rather than die mid-render.
    """
    try:
        from deforum.utils import py3d_tools as p3d
    except Exception as exc:
        print(f"[deforum] splat warp: py3d_tools unavailable ({exc}).")
        return None

    try:
        from .deforum_compat import schedule_value

        translate_xyz = [
            -schedule_value(keys.translation_x_series, frame_idx, 0.0) * TRANSLATION_SCALE,
            schedule_value(keys.translation_y_series, frame_idx, 0.0) * TRANSLATION_SCALE,
            -schedule_value(keys.translation_z_series, frame_idx, 0.0) * TRANSLATION_SCALE,
        ]
        rotate_xyz = [
            math.radians(schedule_value(keys.rotation_3d_x_series, frame_idx, 0.0)),
            math.radians(schedule_value(keys.rotation_3d_y_series, frame_idx, 0.0)),
            math.radians(schedule_value(keys.rotation_3d_z_series, frame_idx, 0.0)),
        ]
        rot_mat = p3d.euler_angles_to_matrix(
            torch.tensor(rotate_xyz, device=device), "XYZ").unsqueeze(0)

        aspect_ratio = float(width) / float(height)
        if not getattr(anim_args, "aspect_ratio_use_old_formula", False):
            aspect_ratio = schedule_value(keys.aspect_ratio_series, frame_idx,
                                          aspect_ratio) or aspect_ratio
        near = schedule_value(keys.near_series, frame_idx, 200.0)
        far = schedule_value(keys.far_series, frame_idx, 10000.0)
        fov_deg = schedule_value(keys.fov_series, frame_idx, 40.0)

        old_cam = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg,
                                            degrees=True, device=device)
        new_cam = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov_deg,
                                            degrees=True, R=rot_mat,
                                            T=torch.tensor([translate_xyz]),
                                            device=device)

        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, dtype=torch.float32, device=device),
            torch.linspace(-1.0, 1.0, width, dtype=torch.float32, device=device),
            indexing="ij")
        z = (torch.ones_like(x) if depth is None
             else torch.as_tensor(depth, dtype=torch.float32, device=device))
        world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

        old_xy = old_cam.get_full_projection_transform().transform_points(world)[:, 0:2]
        new_xy = new_cam.get_full_projection_transform().transform_points(world)[:, 0:2]
        return (new_xy - old_xy).reshape(height, width, 2)
    except Exception as exc:
        print(f"[deforum] splat warp: could not build the camera offset ({exc}); "
              f"leaving the frame unwarped.")
        return None


def splat_warp(image, depth, offset, hole_dilate=2, hole_feather=3):
    """Forward-splat ``image`` by ``offset``, nearest depth winning.

    ``image``  (H, W, 3) float tensor
    ``depth``  (H, W) float tensor, larger = further away, or None
    ``offset`` (H, W, 2) float tensor in normalised [-1, 1] units

    Returns ``(warped, holes)`` where ``holes`` is (H, W) in 0..1, and is 1
    where nothing landed.
    """
    device = image.device
    height, width, channels = image.shape

    # Normalised [-1,1] offsets into pixel offsets. The (n-1)/2 scale is the
    # convention affine_grid uses, so this lines up with deforum's warp.
    dx = offset[..., 0] * (width - 1) / 2.0
    dy = offset[..., 1] * (height - 1) / 2.0

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij")
    # The backward warp samples at (grid - offset), so content moves by
    # (+offset); splatting forward therefore adds it.
    dst_x = torch.round(xs + dx).long()
    dst_y = torch.round(ys + dy).long()

    inside = (dst_x >= 0) & (dst_x < width) & (dst_y >= 0) & (dst_y < height)
    flat_dst = (dst_y.clamp(0, height - 1) * width
                + dst_x.clamp(0, width - 1)).reshape(-1)
    valid = inside.reshape(-1)

    if depth is None:
        z = torch.zeros(height * width, device=device, dtype=torch.float32)
    else:
        z = torch.as_tensor(depth, dtype=torch.float32, device=device).reshape(-1)

    # z-buffer: the nearest sample landing on a destination pixel wins.
    zbuf = torch.full((height * width,), float("inf"), device=device)
    zbuf.scatter_reduce_(0, flat_dst[valid], z[valid], reduce="amin",
                         include_self=True)

    # Average the samples that tie for nearest, which keeps the result stable
    # where a flat surface splats several pixels onto one.
    eps = 1e-4
    winners = valid & (z <= zbuf[flat_dst] + eps)

    accum = torch.zeros(height * width, channels, device=device,
                        dtype=torch.float32)
    counts = torch.zeros(height * width, device=device, dtype=torch.float32)
    flat_src = image.reshape(-1, channels).float()
    accum.index_add_(0, flat_dst[winners], flat_src[winners])
    counts.index_add_(0, flat_dst[winners],
                      torch.ones(int(winners.sum()), device=device))

    filled = counts > 0
    accum[filled] /= counts[filled].unsqueeze(1)
    warped = accum.reshape(height, width, channels)
    holes = (~filled).float().reshape(height, width)

    # Single-pixel gaps are splatting noise rather than real disocclusion, and
    # a hard-edged mask makes the sampler draw a visible seam. Close the former
    # and soften the latter.
    if hole_dilate > 0 or hole_feather > 0:
        import cv2
        holes_np = holes.detach().cpu().numpy()
        if hole_dilate > 0:
            kernel = np.ones((hole_dilate * 2 + 1,) * 2, np.uint8)
            holes_np = cv2.dilate(holes_np, kernel)
        if hole_feather > 0:
            holes_np = cv2.GaussianBlur(holes_np, (0, 0), hole_feather)
        holes = torch.from_numpy(holes_np).to(device)

    holes = holes.clamp(0.0, 1.0)
    # Fill the holes with the unwarped frame so the image is usable on its own.
    # The mask still records which pixels these were.
    hole_weight = holes.unsqueeze(-1)
    warped = warped * (1.0 - hole_weight) + image.float() * hole_weight
    return warped, holes
