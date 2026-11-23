"""Colour matching node with selectable transfer methods.

The existing (deforum) Color Match node histogram-matches against the first
generated frame and offers no alternative. This one exposes the methods in
``modules/deforum_color.py``, so a run can be corrected with the optimal
linear map, or with a wavelet split that leaves the sampler's detail alone,
and can anchor to a running average instead of the first frame.

Drop it where Color Match sits -- after the sampler, before the frame is
cached for the next iteration -- or use it instead of Color Match.
"""

import numpy as np
import torch

from ..modules.deforum_color import (COLOR_ANCHORS, COLOR_METHODS, ColorAnchor,
                                     transfer)


def _to_numpy(image):
    """ComfyUI IMAGE (B,H,W,C float32 0..1) -> a single HWC frame."""
    return image[0].detach().cpu().numpy().astype(np.float32)


def _to_tensor(frame):
    return torch.from_numpy(np.ascontiguousarray(frame)).unsqueeze(0)


class DeforumColorTransferNode:
    """Match a frame's colour to an anchor using a selectable method."""

    def __init__(self):
        self.anchor = None
        self.frames_seen = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (COLOR_METHODS, {"default": "mkl_wavelet"}),
                "anchor": (COLOR_ANCHORS, {"default": "first_frame"}),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,
                                       "step": 0.01}),
                "ema_decay": ("FLOAT", {"default": 0.95, "min": 0.0,
                                        "max": 0.999, "step": 0.001}),
                "legacy_mode": (["LAB", "HSV", "RGB"], {"default": "LAB"}),
                "reset": ("BOOLEAN", {"default": False}),
                "reference": ("IMAGE",),
                "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "match"
    display_name = "Color Transfer"
    CATEGORY = "deforum/image"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # The anchor carries between executions, so this node must never be
        # served from cache.
        return float("NaN")

    def match(self, image, method, anchor, strength=1.0, ema_decay=0.95,
              legacy_mode="LAB", reset=False, reference=None,
              deforum_frame_data=None):
        frame = _to_numpy(image)

        frame_index = (deforum_frame_data or {}).get("frame_idx",
                                                     self.frames_seen)
        # Frame 0 means a new run: a stale anchor from the previous one would
        # drag this run towards the old palette.
        if reset or frame_index == 0 or self.anchor is None:
            self.anchor = ColorAnchor(anchor, ema_decay)
        self.anchor.mode = anchor
        self.anchor.decay = float(ema_decay)

        if reference is not None:
            # An explicit reference overrides the running anchor entirely.
            target = _to_numpy(reference)
        else:
            target = self.anchor.get()

        if target is None:
            self.anchor.update(frame)
            self.frames_seen = frame_index + 1
            return (image,)

        result = transfer(frame, target, method=method, strength=strength,
                          legacy_mode=legacy_mode)
        if reference is None:
            self.anchor.update(result)
        self.frames_seen = frame_index + 1
        return (_to_tensor(result),)
