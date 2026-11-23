"""Colour transfer methods for the deforum feedback loop.

Deforum renders frame N by warping frame N-1 and running img2img over it, so
every colour error is fed back in and compounds. The existing answer in this
pack is ``maintain_colors()``, which histogram-matches against the first
generated frame. It holds the hue but flattens contrast, and over a long run
the match itself drifts, because a per-channel histogram is a weak description
of an image.

The alternatives here are all training-free and run on CPU in a few
milliseconds, so they cost nothing per frame next to a single sampler step:

``reinhard``   mean/std alignment in LAB. Cheap, gentle, keeps contrast.
``mkl``        Monge-Kantorovich linear transport: the closed-form optimal map
               between two Gaussians, so it matches the full covariance rather
               than each channel on its own. Corrects colour *casts* that
               per-channel methods cannot see.
``wavelet``    Low frequencies from the anchor, high frequencies from the new
               frame. Colour lives in the low bands and detail in the high
               ones, so this pins the palette while leaving the sampler's
               texture untouched.

Anchoring matters as much as the method. Matching to the first frame fights the
animation as it legitimately moves away from where it started; matching to the
previous frame lets drift accumulate freely. ``ema`` keeps a running anchor
that follows slow intentional change while resisting per-frame wander.
"""

import cv2
import numpy as np

COLOR_METHODS = ["none", "legacy", "reinhard", "mkl", "wavelet", "mkl_wavelet"]
COLOR_ANCHORS = ["first_frame", "previous", "ema"]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _as_float(image):
    """uint8 RGB -> float32 RGB in 0..1, remembering whether to convert back."""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0, True
    return image.astype(np.float32), False


def _restore(image, was_uint8):
    image = np.clip(image, 0.0, 1.0)
    if was_uint8:
        return (image * 255.0).round().astype(np.uint8)
    # float32 explicitly: the matrix maths in mkl() promotes to float64, which
    # then refuses to go through cv2 colour conversions and is not what
    # ComfyUI hands around either.
    return image.astype(np.float32)


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------

def reinhard(source, anchor):
    """Align per-channel mean and std in LAB (Reinhard et al., 2001)."""
    src = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
    ref = cv2.cvtColor(anchor, cv2.COLOR_RGB2LAB)
    out = np.empty_like(src)
    for channel in range(3):
        s, r = src[..., channel], ref[..., channel]
        s_std = s.std()
        # A flat channel has no distribution to match, and scaling it by
        # ref_std / 0 would blow the frame out.
        scale = (r.std() / s_std) if s_std > 1e-5 else 1.0
        out[..., channel] = (s - s.mean()) * scale + r.mean()
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def _sqrtm_psd(matrix):
    """Square root of a small symmetric positive-semidefinite matrix."""
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def mkl(source, anchor):
    """Monge-Kantorovich linear colour transport (Pitie & Kokaram, 2007).

    The optimal linear map between two Gaussians. This is what lets it fix
    casts that per-channel matching leaves behind: it uses the covariance
    between channels, not just each channel's spread.
    """
    src = source.reshape(-1, 3)
    ref = anchor.reshape(-1, 3)
    src_mean, ref_mean = src.mean(0), ref.mean(0)
    src_cov = np.cov(src, rowvar=False) + np.eye(3) * 1e-6
    ref_cov = np.cov(ref, rowvar=False) + np.eye(3) * 1e-6

    src_root = _sqrtm_psd(src_cov)
    src_root_inv = np.linalg.inv(src_root)
    transform = (src_root_inv
                 @ _sqrtm_psd(src_root @ ref_cov @ src_root)
                 @ src_root_inv)

    return ((src - src_mean) @ transform.T + ref_mean).reshape(source.shape)


def _wavelet_blur(image, radius):
    """Blur with a dilated 3x3 kernel: one level of a stationary wavelet."""
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16.0
    # cv2 has no dilation argument, so build the spread kernel explicitly.
    size = 2 * radius + 1
    dilated = np.zeros((size, size), np.float32)
    dilated[::radius, ::radius] = kernel
    return cv2.filter2D(image, -1, dilated, borderType=cv2.BORDER_REFLECT)


def _wavelet_split(image, levels=5):
    """Separate accumulated detail from the residual low-frequency image."""
    high = np.zeros_like(image)
    current = image
    for level in range(levels):
        low = _wavelet_blur(current, 2 ** level)
        high += current - low
        current = low
    return high, current


def wavelet(source, anchor, levels=5):
    """Detail from the new frame, colour from the anchor."""
    source_high, _ = _wavelet_split(source, levels)
    _, anchor_low = _wavelet_split(anchor, levels)
    return source_high + anchor_low


def mkl_wavelet(source, anchor, levels=5):
    """MKL for the cast, wavelet to hand the detail back.

    Measured separately, mkl controls the palette best but costs a little
    texture, while wavelet preserves texture exactly but controls the palette
    less tightly. Running mkl first and then taking its low frequencies with
    the original frame's high frequencies gets both: the colour map is the
    optimal linear one, and no detail is spent achieving it.
    """
    corrected = mkl(source, anchor)
    source_high, _ = _wavelet_split(source, levels)
    _, corrected_low = _wavelet_split(corrected, levels)
    return source_high + corrected_low


def legacy(source, anchor, mode="LAB"):
    """The pack's existing behaviour, for a like-for-like comparison.

    deforum's ``maintain_colors`` takes and returns BGR while the reference
    sample stays RGB. Getting that wrong is what made Color Match tint
    everything teal (fixed in 9485d53), so it is wrapped rather than called
    directly.
    """
    from deforum.utils.image_utils import maintain_colors
    src = cv2.cvtColor(_restore(source, True), cv2.COLOR_RGB2BGR)
    ref = _restore(anchor, True)
    matched = maintain_colors(src, ref, mode)
    rgb = cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 255.0


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def transfer(source, anchor, method="wavelet", strength=1.0, legacy_mode="LAB"):
    """Pull ``source`` towards ``anchor``'s colour.

    ``strength`` blends the result back towards the input, so 0 is a no-op and
    values below 1 correct partially, which is useful when a full correction
    fights an intentional lighting change.
    """
    if method == "none" or strength <= 0.0 or anchor is None:
        return source

    source_f, was_uint8 = _as_float(source)
    anchor_f, _ = _as_float(anchor)

    if method == "reinhard":
        result = reinhard(source_f, anchor_f)
    elif method == "mkl":
        result = mkl(source_f, anchor_f)
    elif method == "wavelet":
        result = wavelet(source_f, anchor_f)
    elif method == "mkl_wavelet":
        result = mkl_wavelet(source_f, anchor_f)
    elif method == "legacy":
        result = legacy(source_f, anchor_f, legacy_mode)
    else:
        raise ValueError(f"unknown colour method {method!r}")

    if strength < 1.0:
        result = source_f * (1.0 - strength) + result * strength
    return _restore(result, was_uint8)


class ColorAnchor:
    """Holds what each frame is matched against.

    ``ema`` is the reason this class exists: a running average follows a scene
    that is genuinely changing while damping the per-frame wander that makes a
    long deforum run slide into a single hue.
    """

    def __init__(self, mode="first_frame", decay=0.9):
        self.mode = mode
        self.decay = float(decay)
        self.anchor = None

    def get(self):
        return self.anchor

    def update(self, frame):
        frame_f, _ = _as_float(frame)
        if self.anchor is None:
            self.anchor = frame_f
            return
        if self.mode == "first_frame":
            return
        if self.mode == "previous":
            self.anchor = frame_f
        elif self.mode == "ema":
            self.anchor = self.decay * self.anchor + (1.0 - self.decay) * frame_f

    def reset(self):
        self.anchor = None
