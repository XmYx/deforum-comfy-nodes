"""Compatibility helpers shared by the deforum nodes.

Everything in here exists so the node pack keeps working across ComfyUI
releases, torch backends (CUDA / ROCm / MPS / CPU) and the moviepy 1.x -> 2.x
split, instead of patching ComfyUI internals.
"""

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils

try:  # ComfyUI moved ExecutionBlocker around a few times
    from comfy_execution.graph import ExecutionBlocker
except ImportError:  # pragma: no cover - very old ComfyUI
    try:
        from comfy_execution.graph_utils import ExecutionBlocker
    except ImportError:
        ExecutionBlocker = None


def block(reason=None):
    """Return a value that silently stops downstream nodes from executing.

    Deforum's cache nodes have nothing to hand out on the first frame of a run.
    Historically the pack returned ``None`` and monkeypatched ComfyUI's executor
    so those ``None``s were treated as "skip this node". ComfyUI has supported
    this natively via ExecutionBlocker for a long time, so use that instead.
    """
    if ExecutionBlocker is None:
        return None
    return ExecutionBlocker(reason)


def synchronize():
    """Flush the active accelerator, whatever it happens to be.

    ``torch.cuda.synchronize()`` raises on Apple MPS and on CPU-only builds, and
    used to take down the whole run on those machines.
    """
    try:
        device = comfy.model_management.get_torch_device()
        device_type = getattr(device, "type", str(device))
    except Exception:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device_type == "mps" and getattr(torch.backends, "mps", None) is not None:
            torch.mps.synchronize()
        elif device_type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize()
    except Exception as exc:  # never let a sync call kill a render
        print(f"[deforum] device synchronize skipped: {exc}")


# --------------------------------------------------------------------------
# Latent geometry
# --------------------------------------------------------------------------

# Fallbacks for when the user has not wired a model into the iterator.
# (latent_channels, spatial_downscale_ratio)
LATENT_SPECS = {
    "stable_diffusion": (4, 8),
    "sdxl": (4, 8),
    "stable_cascade": (16, 42),
    "sd3": (16, 8),
    "flux": (16, 8),          # FLUX.1
    "flux2": (128, 16),       # FLUX.2 / FLUX.3 share this latent format
    "flux3": (128, 16),
    "qwen_image": (16, 8),
    "krea2": (16, 8),
}

LATENT_TYPES = ["auto"] + list(LATENT_SPECS)


def latent_geometry(latent_type="auto", model=None):
    """Return ``(channels, downscale)`` for the initial deforum latent.

    With a model wired in we ask ComfyUI for the real latent format, which is
    what makes new architectures work without this pack being updated. The
    dropdown stays as a manual override / fallback.
    """
    if model is not None and latent_type == "auto":
        try:
            latent_format = model.get_model_object("latent_format")
            return (
                int(latent_format.latent_channels),
                int(latent_format.spacial_downscale_ratio),
            )
        except Exception as exc:
            print(f"[deforum] could not read latent format from model ({exc}), falling back to SD defaults")

    if latent_type == "auto":
        latent_type = "stable_diffusion"
    return LATENT_SPECS.get(latent_type, LATENT_SPECS["stable_diffusion"])


# --------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------

def schedule_series(schedule, max_frames, seed=-1, fps=24):
    """Parse any deforum schedule string into a per-frame pandas Series.

    ``"0:(0), 30:(1)"`` over 60 frames gives a ramp from 0 to 1 and then holds.
    This is the same parser the built-in schedules use, so expressions such as
    ``0:(0.5+0.5*sin(3.14*t/30))`` work here too.
    """
    from types import SimpleNamespace

    from deforum.pipelines.deforum_animation.animation_helpers import FrameInterpolator

    max_frames = max(1, int(max_frames))
    settings = SimpleNamespace(max_frames=max_frames, fps=fps)
    interpolator = FrameInterpolator(settings, seed)
    return interpolator.get_inbetweens(interpolator.parse_key_frames(str(schedule)))


def schedule_value(series, index, default=None):
    """Read ``series[index]``, clamped to the series it actually has.

    DeforumAnimKeys builds its schedule series from ``max_frames``, but cadence
    walks tween frames that can land past the last frame, and a restarted run
    can ask for an index from the previous, longer animation. Indexing straight
    into the pandas Series raised ``KeyError`` and killed the render near the
    end of a run.
    """
    if series is None:
        return default
    try:
        length = len(series)
    except TypeError:
        return default
    if length == 0:
        return default
    try:
        i = int(index)
    except (TypeError, ValueError):
        return default
    i = min(max(i, 0), length - 1)
    try:
        return series[i]
    except (KeyError, IndexError):
        try:
            return series.iloc[i]
        except Exception:
            return default


# --------------------------------------------------------------------------
# Conditioning
# --------------------------------------------------------------------------

def encode_prompt(clip, prompt):
    """Encode ``prompt`` the way ComfyUI's own CLIPTextEncode does.

    Going through ``encode_from_tokens_scheduled`` is what gives us SD3.x, FLUX
    and Qwen-Image support: those text encoders return extra conditioning keys
    (``guidance``, multiple pooled outputs) that the old
    ``encode_from_tokens(return_pooled=True)`` call silently dropped.
    """
    if clip is None:
        raise RuntimeError(
            "ERROR: clip input is invalid: None\n\n"
            "The checkpoint you loaded does not contain a text encoder. "
            "SD3.x, FLUX and Qwen-Image checkpoints usually ship the text encoders "
            "separately - load them with a (Dual)CLIPLoader and feed that CLIP into "
            "the deforum conditioning node."
        )

    tokens = clip.tokenize(prompt)
    if hasattr(clip, "encode_from_tokens_scheduled"):
        return clip.encode_from_tokens_scheduled(tokens)

    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def resolve_sampler_name(name, default="euler"):
    """Map a deforum / A1111 sampler name onto a ComfyUI one."""
    aliases = {
        "euler_a": "euler_ancestral",
        "euler a": "euler_ancestral",
        "euler ancestral": "euler_ancestral",
        "dpm++ 2m": "dpmpp_2m",
        "dpm++ 2m karras": "dpmpp_2m",
        "dpm++ sde": "dpmpp_sde",
        "dpm++ 2s a": "dpmpp_2s_ancestral",
        "lms": "lms",
        "heun": "heun",
        "ddim": "ddim",
        "plms": "euler",
    }
    if not name:
        return default
    if name in comfy.samplers.KSampler.SAMPLERS:
        return name
    resolved = aliases.get(str(name).strip().lower())
    if resolved in comfy.samplers.KSampler.SAMPLERS:
        return resolved
    print(f"[deforum] unknown sampler '{name}', falling back to '{default}'")
    return default


def resolve_scheduler_name(name, default="normal"):
    if not name:
        return default
    if name in comfy.samplers.KSampler.SCHEDULERS:
        return name
    print(f"[deforum] unknown scheduler '{name}', falling back to '{default}'")
    return default


# --------------------------------------------------------------------------
# moviepy 1.x / 2.x
# --------------------------------------------------------------------------

def get_moviepy():
    """Return the moviepy namespace regardless of major version.

    moviepy 2.0 dropped the ``moviepy.editor`` shim; importing it was the single
    most common install failure reported against this pack.
    """
    try:
        import moviepy.editor as mp  # moviepy 1.x
        return mp
    except ImportError:
        pass
    try:
        import moviepy as mp  # moviepy 2.x re-exports at the top level
        if hasattr(mp, "ImageSequenceClip"):
            return mp
    except ImportError:
        pass
    raise ImportError(
        "moviepy is required by the deforum video nodes. Install it with "
        "`pip install \"moviepy>=1.0.3\"` into the same Python environment that runs ComfyUI."
    )


def clip_subclip(clip, start, end):
    """``subclip`` was renamed to ``subclipped`` in moviepy 2.x."""
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start, end)
    return clip.subclip(start, end)


def clip_with_audio(clip, audio_clip):
    """``set_audio`` was renamed to ``with_audio`` in moviepy 2.x."""
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio_clip)
    return clip.set_audio(audio_clip)
