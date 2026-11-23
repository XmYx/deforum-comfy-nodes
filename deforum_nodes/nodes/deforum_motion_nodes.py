"""Depth, warp and cadence nodes built on the newer methods.

All three sit beside the existing nodes rather than replacing them, so an
existing workflow behaves exactly as before until one is wired in.

The important one is Splat Warp 3D: it emits a *disocclusion mask* alongside
the warped frame. Feed that mask through SetLatentNoiseMask and the sampler
can regenerate only where the camera revealed new area, instead of the whole
frame getting the same ``strength_schedule`` value. That is the thing a single
global denoise strength cannot express.
"""

import numpy as np
import torch

from ..modules.deforum_cadence_fx import FLOW_METHODS, blend
from ..modules.deforum_depth import (DEPTH_ANYTHING_ALGORITHMS,
                                     DeforumDepthPredictor, DepthEMA)
from ..modules.deforum_warp import camera_offset, splat_warp


def _image_to_numpy(image):
    return image[0].detach().cpu().numpy().astype(np.float32)


def _numpy_to_image(array):
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def _frame_index(deforum_frame_data, fallback=0):
    return (deforum_frame_data or {}).get("frame_idx", fallback)


class DeforumDepthAnythingNode:
    """Predict depth with Depth Anything V2, optionally averaged over time."""

    def __init__(self):
        self.predictor = None
        self.model_name = None
        self.size = None
        self.ema = DepthEMA(0.0)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (DEPTH_ANYTHING_ALGORITHMS,
                          {"default": "Depth-Anything-V2-Small"}),
            },
            "optional": {
                "temporal_decay": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.95, "step": 0.01,
                    "tooltip": "Average with the previous frame's depth. 0 is "
                               "off. Around 0.5 removes most 3D shimmer."}),
                "match_legacy_scale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Map the depth onto the range deforum's MiDaS "
                               "path produces, so existing camera schedules "
                               "keep their meaning when you switch model."}),
                "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
            },
        }

    RETURN_TYPES = ("IMAGE", "DEPTH_MAP")
    RETURN_NAMES = ("depth_preview", "depth")
    FUNCTION = "predict"
    display_name = "Depth Anything V2"
    CATEGORY = "deforum/depth"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def predict(self, image, model, temporal_decay=0.0,
                match_legacy_scale=True, deforum_frame_data=None):
        frame = _image_to_numpy(image)
        height, width = frame.shape[0], frame.shape[1]
        if (self.predictor is None or self.model_name != model
                or self.size != (width, height)
                or self.predictor.match_legacy_scale != match_legacy_scale):
            self.predictor = DeforumDepthPredictor(
                model, width=width, height=height,
                match_legacy_scale=match_legacy_scale)
            self.model_name = model
            self.size = (width, height)

        frame_index = _frame_index(deforum_frame_data)
        if frame_index == 0:
            self.ema.reset()
        self.ema.decay = float(temporal_decay)

        depth = self.predictor.predict(
            (np.clip(frame, 0.0, 1.0) * 255.0).round().astype(np.uint8))

        offset = None
        if temporal_decay > 0 and deforum_frame_data:
            keys = deforum_frame_data.get("keys")
            anim_args = deforum_frame_data.get("anim_args")
            if keys is not None and anim_args is not None:
                offset = camera_offset(depth, anim_args, keys, frame_index,
                                       frame.shape[1], frame.shape[0], "cpu")
        depth = self.ema.apply(depth, offset)

        # A 0..1 greyscale copy so the map can be previewed or saved.
        low, high = float(depth.min()), float(depth.max())
        spread = (high - low) if (high - low) > 1e-6 else 1.0
        preview = ((depth - low) / spread).unsqueeze(-1).repeat(1, 1, 3)
        return (preview.unsqueeze(0), depth)


class DeforumSplatWarpNode:
    """Forward-splatting 3D warp that also reports where content is new."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
            },
            "optional": {
                "depth": ("DEPTH_MAP",),
                "hole_dilate": ("INT", {
                    "default": 2, "min": 0, "max": 32,
                    "tooltip": "Close single-pixel splatting gaps, which are "
                               "noise rather than real disocclusion."}),
                "hole_feather": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 32.0, "step": 0.5,
                    "tooltip": "Soften the mask edge so the sampler does not "
                               "leave a visible seam."}),
                "mask_gain": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                    "tooltip": "Scale the mask before it leaves this node."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "disocclusion_mask")
    FUNCTION = "warp"
    display_name = "Splat Warp 3D"
    CATEGORY = "deforum/warp"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def warp(self, image, deforum_frame_data, depth=None, hole_dilate=2,
             hole_feather=3.0, mask_gain=1.0):
        frame = image[0].detach().cpu().float()
        height, width = frame.shape[0], frame.shape[1]

        keys = (deforum_frame_data or {}).get("keys")
        anim_args = (deforum_frame_data or {}).get("anim_args")
        frame_index = _frame_index(deforum_frame_data)

        offset = None
        if keys is not None and anim_args is not None:
            offset = camera_offset(depth, anim_args, keys, frame_index,
                                   width, height, "cpu")
        if offset is None:
            # Nothing to warp by: hand the frame back with an empty mask rather
            # than guessing at a transform.
            empty = torch.zeros((1, height, width), dtype=torch.float32)
            return (image, empty)

        warped, holes = splat_warp(frame, depth, offset,
                                   hole_dilate=int(hole_dilate),
                                   hole_feather=float(hole_feather))
        if mask_gain != 1.0:
            holes = (holes * float(mask_gain)).clamp(0.0, 1.0)
        return (warped.unsqueeze(0), holes.unsqueeze(0))


class DeforumOcclusionCadenceNode:
    """Tween two frames along optical flow, without ghosting occlusions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "t": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0,
                                "step": 0.01}),
            },
            "optional": {
                "method": (FLOW_METHODS, {"default": "DIS Medium"}),
                "tolerance": ("FLOAT", {
                    "default": 1.5, "min": 0.1, "max": 16.0, "step": 0.1,
                    "tooltip": "How far a round-tripped pixel may land from "
                               "where it started before the flow is distrusted."}),
                "feather": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 16.0,
                                      "step": 0.5}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "tween"
    display_name = "Occlusion Aware Cadence"
    CATEGORY = "deforum/cadence"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def tween(self, image_a, image_b, t, method="DIS Medium", tolerance=1.5,
              feather=2.0):
        a = _image_to_numpy(image_a)
        b = _image_to_numpy(image_b)
        if a.shape != b.shape:
            raise ValueError(
                f"Occlusion Aware Cadence needs matching sizes, got "
                f"{a.shape[1]}x{a.shape[0]} and {b.shape[1]}x{b.shape[0]}")
        result = blend(a, b, t, method=method, tolerance=tolerance,
                       feather=feather)
        return (_numpy_to_image(result),)
