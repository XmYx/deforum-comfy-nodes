"""Depth for the 3D warp: newer models, and a temporal average.

Two things are going on here.

**Model choice.** The depth models deforum's dropdown offered (AdaBins, MiDaS,
Leres, Zoe) are 2019-2022, and MiDaS in particular gives soft, blobby
boundaries. Since the 3D warp displaces every pixel by its depth, a soft
boundary becomes a smeared edge. Depth Anything V2 is much sharper at
comparable speed, and it turns out ``deforum.models.DepthModel`` already knows
how to load it -- any ``depth_algorithm`` starting with ``depth-anything``
builds the HF model, picking Base when the name contains "base" and Small
otherwise. It was simply missing from this pack's dropdown, which is fixed in
``deforum_ui_data.py``.

Going through ``DepthModel`` rather than calling transformers directly is
deliberate: the warp treats depth as a world z coordinate, and deforum feeds
MiDaS's raw inverse depth straight in, so its values land around 15-21 and
*larger means nearer*. A hand-rolled predictor normalised to 0..1 puts every
point essentially on the camera plane and the projection explodes -- measured,
not guessed: offsets of 66 px and 52% of the frame reported as disocclusion on
a still frame, against 0.0 px and 0% once the scale matched. Inheriting the
convention is the only sane option.

**Temporal stability.** Most of deforum's 3D shimmer is not the warp, it is the
depth map changing every frame where the image barely changed. A video depth
model is the principled fix but needs future frames, which an autoregressive
loop does not have. An exponential moving average costs nothing and attacks the
same artefact. ``DepthModel`` has no such thing, which is why it lives here.
"""

import torch

# Added to the depth_algorithm dropdown. Both are Apache-2.0; ViT-L is
# CC-BY-NC and is deliberately not offered.
DEPTH_ANYTHING_ALGORITHMS = [
    "Depth-Anything-V2-Small",
    "Depth-Anything-V2-Base",
]

# The warp uses depth directly as a world z coordinate, so its absolute scale
# decides how much parallax a given translation produces. deforum's MiDaS path
# lands around 15-21; its depth-anything branch returns raw predicted_depth,
# which measured 1.6-3.3 on the same frame. Switching model would therefore
# multiply the effective camera motion by roughly seven and quietly invalidate
# every schedule in a saved workflow. Mapping each algorithm onto the same
# working range keeps a schedule meaning what it meant.
LEGACY_DEPTH_RANGE = (14.9, 20.9)


def rescale_to_legacy_range(depth, span=LEGACY_DEPTH_RANGE):
    """Affinely map depth onto deforum's MiDaS-like range.

    Uses the 2nd/98th percentiles rather than min/max so one speck of outlier
    depth cannot squash the whole frame.
    """
    flat = depth.flatten().float()
    low = torch.quantile(flat, 0.02)
    high = torch.quantile(flat, 0.98)
    if float(high - low) < 1e-6:
        return torch.full_like(depth, float(sum(span)) / 2.0)
    normalised = (depth - low) / (high - low)
    return normalised * (span[1] - span[0]) + span[0]


class DeforumDepthPredictor:
    """Thin wrapper over deforum's DepthModel, cached per algorithm.

    Kept module-level so switching algorithms in a workflow does not reload
    the previous one on every frame.
    """

    _cache = {}

    def __init__(self, algorithm="Depth-Anything-V2-Small", width=512,
                 height=512, midas_weight=0.2, device=None,
                 match_legacy_scale=True):
        self.algorithm = algorithm
        self.match_legacy_scale = match_legacy_scale
        self.width = width
        self.height = height
        self.midas_weight = midas_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _model(self):
        key = (self.algorithm, self.width, self.height, self.device)
        model = DeforumDepthPredictor._cache.get(key)
        if model is None:
            from deforum.models import DepthModel
            from deforum.utils import constants

            model = DepthModel(constants.config.model_dir, self.device, True,
                               keep_in_vram=True,
                               depth_algorithm=self.algorithm,
                               Width=self.width, Height=self.height,
                               midas_weight=self.midas_weight)
            DeforumDepthPredictor._cache[key] = model
            print(f"[deforum] depth model ready: {self.algorithm}")
        return model

    def predict(self, image_rgb_u8):
        """``image_rgb_u8`` is (H, W, 3) uint8 RGB. Returns (H, W) float32.

        Values follow deforum's own convention, whatever the algorithm: this
        is the tensor its 3D warp expects as z.
        """
        depth = self._model().predict(image_rgb_u8, self.midas_weight, True)
        depth = torch.as_tensor(depth).float().cpu()
        if self.match_legacy_scale:
            depth = rescale_to_legacy_range(depth)
        return depth


class DepthEMA:
    """Running average of successive depth maps.

    ``decay`` 0 disables it, which is how the node defaults, so nothing
    changes unless it is asked for.

    The previous map is resampled by the camera offset before averaging where
    one is supplied -- otherwise the average smears depth along the motion path
    instead of stabilising it.
    """

    def __init__(self, decay=0.0):
        self.decay = float(decay)
        self.previous = None

    def reset(self):
        self.previous = None

    def apply(self, depth, offset=None):
        if self.decay <= 0.0:
            self.previous = depth
            return depth
        if self.previous is None or self.previous.shape != depth.shape:
            self.previous = depth
            return depth

        aligned = self.previous
        if offset is not None:
            aligned = self._align(self.previous, offset)
        blended = self.decay * aligned + (1.0 - self.decay) * depth
        self.previous = blended
        return blended

    @staticmethod
    def _align(depth, offset):
        """Resample the previous depth into this frame using the camera offset."""
        height, width = depth.shape
        device = depth.device
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device),
            torch.linspace(-1.0, 1.0, width, device=device),
            indexing="ij")
        grid = torch.stack((xs, ys), dim=-1) - offset.to(device)
        sampled = torch.nn.functional.grid_sample(
            depth.unsqueeze(0).unsqueeze(0), grid.unsqueeze(0),
            mode="bilinear", padding_mode="border", align_corners=False)
        return sampled[0, 0]
