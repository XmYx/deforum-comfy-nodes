"""Scheduled guide images.

A1111's deforum lets you pin an init image to a keyframe and keep pulling the
animation towards it. The Iterator's ``init_latent`` input already covers
seeding frame 0; what was missing is the *ongoing* influence, which is what
these nodes add.
"""

import torch

import comfy.utils

from ..modules.deforum_compat import schedule_series, schedule_value


def _resize_to(image, height, width):
    """Resize an ``[H, W, C]`` image tensor to match the working frame."""
    if image.shape[0] == height and image.shape[1] == width:
        return image
    chw = image.permute(2, 0, 1).unsqueeze(0)
    chw = comfy.utils.common_upscale(chw, width, height, "lanczos", "center")
    return chw.squeeze(0).permute(1, 2, 0)


class DeforumGuideImageNode:
    """Blend the current frame towards a guide image chosen by schedule."""

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Deforum nodes re-run every frame, so always report a new hash.
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "The frame being carried around the deforum loop, "
                               "normally straight out of Frame Warp.",
                }),
                "guide_images": ("IMAGE", {
                    "tooltip": "One or more guide images. Batch several together "
                               "and pick between them with image_schedule.",
                }),
                "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                "strength_schedule": ("STRING", {
                    "default": "0:(0.5), 2:(0.0)",
                    "tooltip": "How strongly each frame is pulled towards the guide. "
                               "0 passes the frame through untouched, 1 replaces it. "
                               "This node sits in the feedback loop, so a strength "
                               "held across many frames COMPOUNDS and the animation "
                               "converges onto the guide. Prefer pulses that return "
                               "to 0, e.g. 0:(1.0), 2:(0.0), 58:(0.0), 60:(0.6), "
                               "62:(0.0) -- strong at each keyframe, free in between. "
                               "A sustained value only works very low (~0.02-0.05).",
                }),
                "image_schedule": ("STRING", {
                    "default": "0:(0)",
                    "tooltip": "Which guide image to use, by index into the batch. "
                               "e.g. 0:(0), 30:(1), 60:(2). Fractional values "
                               "crossfade between neighbouring guides.",
                }),
                "crossfade_guides": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Blend between neighbouring guide images while the "
                               "schedule moves between them, instead of snapping.",
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional. Restrict the guide to part of the frame.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "FLOAT")
    RETURN_NAMES = ("IMAGE", "GUIDE", "STRENGTH")
    FUNCTION = "fn"
    display_name = "Guide Image Schedule"
    CATEGORY = "deforum/image"

    def __init__(self):
        self._cache_key = None
        self._strength = None
        self._index = None

    def _series(self, deforum_frame_data, strength_schedule, image_schedule):
        anim_args = deforum_frame_data.get("anim_args")
        max_frames = int(getattr(anim_args, "max_frames", 0) or 0)
        # The schedules only need parsing when they or the length change.
        key = (strength_schedule, image_schedule, max_frames)
        if key != self._cache_key:
            # +1 so the last frame can still be read without clamping.
            self._strength = schedule_series(strength_schedule, max_frames + 1)
            self._index = schedule_series(image_schedule, max_frames + 1)
            self._cache_key = key
        return self._strength, self._index

    def _guide_for(self, guide_images, index, crossfade):
        count = guide_images.shape[0]
        index = min(max(float(index), 0.0), float(count - 1))
        low = int(index)
        if not crossfade or low >= count - 1:
            return guide_images[min(low, count - 1)]
        t = index - low
        if t <= 0.0:
            return guide_images[low]
        return guide_images[low] * (1.0 - t) + guide_images[low + 1] * t

    @torch.inference_mode()
    def fn(self, image, guide_images, deforum_frame_data, strength_schedule,
           image_schedule, crossfade_guides, mask=None):
        if image is None or guide_images is None or guide_images.shape[0] == 0:
            return (image, guide_images, 0.0)

        frame_idx = deforum_frame_data.get("frame_idx", 0) or 0
        strength_series, index_series = self._series(
            deforum_frame_data, strength_schedule, image_schedule)

        strength = float(schedule_value(strength_series, frame_idx, 0.0) or 0.0)
        strength = min(max(strength, 0.0), 1.0)
        index = schedule_value(index_series, frame_idx, 0.0) or 0.0

        frame = image[0]
        guide = self._guide_for(guide_images, index, crossfade_guides)
        guide = _resize_to(guide.to(frame.device, frame.dtype),
                           frame.shape[0], frame.shape[1])

        if strength <= 0.0:
            return (image, guide.unsqueeze(0), 0.0)

        alpha = torch.full_like(frame[..., :1], strength)
        if mask is not None and mask.numel():
            m = mask[0] if mask.dim() == 3 else mask
            m = _resize_to(m.unsqueeze(-1).to(frame.device, frame.dtype),
                           frame.shape[0], frame.shape[1])
            alpha = alpha * m

        blended = frame * (1.0 - alpha) + guide * alpha
        print(f"[deforum] Guide image: index {index:.2f}, strength {strength:.3f}")
        return (blended.unsqueeze(0), guide.unsqueeze(0), strength)


class DeforumGuideImageScheduleValueNode:
    """Read any deforum schedule as a number for the current frame."""

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Deforum nodes re-run every frame, so always report a new hash.
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                "schedule": ("STRING", {
                    "default": "0:(1.0)",
                    "tooltip": "Any deforum schedule, e.g. 0:(1.0), 30:(0.2). "
                               "Useful for driving a ControlNet or IPAdapter "
                               "strength from the deforum timeline.",
                }),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("FLOAT", "INT")
    FUNCTION = "fn"
    display_name = "Schedule Value"
    CATEGORY = "deforum/parameters"

    def __init__(self):
        self._cache_key = None
        self._series = None

    def fn(self, deforum_frame_data, schedule):
        anim_args = deforum_frame_data.get("anim_args")
        max_frames = int(getattr(anim_args, "max_frames", 0) or 0)
        key = (schedule, max_frames)
        if key != self._cache_key:
            self._series = schedule_series(schedule, max_frames + 1)
            self._cache_key = key
        value = float(schedule_value(self._series, deforum_frame_data.get("frame_idx", 0), 0.0) or 0.0)
        return (value, int(round(value)))
