"""Occlusion-aware blending for cadence tween frames.

The existing cadence warps between two sampled keyframes along optical flow.
Where the flow is wrong -- and it is always wrong somewhere, because occluded
pixels have no correspondence to find -- blending the two warps produces a
ghost: two faint copies of a moving edge.

Forward-backward consistency finds those regions for free. Follow a pixel's
forward flow, then the backward flow where it lands, and it should return to
where it started; the distance by which it fails to is the classic occlusion
detector. Those pixels get taken from the nearer keyframe instead of blended,
which trades a ghost for a slightly stale but coherent edge.

No model and no download. Flow computation and flow warping are reused from
``interp.py`` rather than reimplemented, so this follows whatever method the
rest of the pack is already using.
"""

import cv2
import numpy as np

from .interp import get_flow_from_images, image_transform_optical_flow

# The method strings get_flow_from_images() understands.
FLOW_METHODS = ["DIS Medium", "DIS Fine", "Farneback Normal", "Farneback Fine"]


def _to_bgr_u8(image):
    """(H, W, 3) float 0..1 RGB -> uint8 BGR, which interp.py expects."""
    return cv2.cvtColor((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def consistency_mask(forward, backward, tolerance=1.5):
    """1 where the two flows agree, 0 where they contradict each other."""
    height, width = forward.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    sampled = cv2.remap(backward,
                        grid_x + forward[..., 0], grid_y + forward[..., 1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    residual = forward + sampled          # should cancel out to zero
    error = np.linalg.norm(residual, axis=-1)
    return (error <= float(tolerance)).astype(np.float32)


def blend(previous, nxt, t, method="DIS Medium", tolerance=1.5, feather=2.0):
    """Tween between two frames at ``t`` in 0..1, avoiding ghosted occlusions.

    ``previous`` and ``nxt`` are (H, W, 3) float 0..1 RGB, and the result is
    the same.
    """
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.0:
        return np.array(previous, copy=True)
    if t >= 1.0:
        return np.array(nxt, copy=True)

    previous_bgr = _to_bgr_u8(previous)
    next_bgr = _to_bgr_u8(nxt)
    forward = get_flow_from_images(previous_bgr, next_bgr, method)
    backward = get_flow_from_images(next_bgr, previous_bgr, method)

    # Warp each endpoint part of the way towards the other.
    warped_previous = image_transform_optical_flow(
        np.array(previous, dtype=np.float32), forward * t)
    warped_next = image_transform_optical_flow(
        np.array(nxt, dtype=np.float32), backward * (1.0 - t))

    reliable = consistency_mask(forward, backward, tolerance)
    if feather > 0:
        reliable = cv2.GaussianBlur(reliable, (0, 0), float(feather))
    reliable = np.clip(reliable, 0.0, 1.0)[..., None]

    blended = warped_previous * (1.0 - t) + warped_next * t
    # Where the flows disagree, trust the nearer keyframe rather than average
    # two warps that do not describe the same surface.
    fallback = warped_previous if t < 0.5 else warped_next
    result = blended * reliable + fallback * (1.0 - reliable)
    return np.clip(result, 0.0, 1.0).astype(np.float32)
