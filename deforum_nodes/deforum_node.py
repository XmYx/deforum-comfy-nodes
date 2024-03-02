# import torch
# import contextlib
# import os
# import math
#
# import comfy.utils
# import comfy.model_management
# from comfy.clip_vision import clip_preprocess
# from comfy.ldm.modules.attention import optimized_attention
# import folder_paths
import copy
import importlib
import inspect
import json
import math
import os
import random
import re
import secrets
import time
from types import SimpleNamespace

import imageio
from tqdm import tqdm

import folder_paths
import hashlib
import cv2
import numexpr
import numpy as np
import pandas as pd
import torch
from PIL import Image

from deforum import DeforumAnimationPipeline, ImageRNGNoise, FilmModel
from deforum.generators.deforum_flow_generator import get_flow_from_images
from deforum.generators.deforum_noise_generator import add_noise
from deforum.generators.rng_noise_generator import slerp
from deforum.models import RAFT, DepthModel
from deforum.pipeline_utils import next_seed
from deforum.pipelines.deforum_animation.animation_helpers import DeforumAnimKeys
from deforum.pipelines.deforum_animation.animation_params import RootArgs, DeforumArgs, DeforumAnimArgs, \
    DeforumOutputArgs, LoopArgs, ParseqArgs
from deforum.pipelines.deforum_animation.pipeline_deforum_animation import interpolate_areas
from deforum.utils.image_utils import maintain_colors, unsharp_mask, compose_mask_with_check, \
    image_transform_optical_flow
from deforum.utils.string_utils import substitute_placeholders, split_weighted_subprompts
from nodes import MAX_RESOLUTION
from .deforum_ui_data import (deforum_base_params, deforum_anim_params, deforum_translation_params,
                              deforum_cadence_params, deforum_masking_params, deforum_depth_params,
                              deforum_noise_params, deforum_color_coherence_params, deforum_diffusion_schedule_params,
                              deforum_hybrid_video_params, deforum_video_init_params, deforum_image_init_params,
                              deforum_hybrid_video_schedules)
from .deforum_node_base import DeforumDataBase
import torch.nn.functional as F
import comfy
deforum_cache = {}
deforum_models = {}
video_extensions = ['webm', 'mp4', 'mkv', 'gif']
deforum_depth_algo = ""

from .standalone_cadence import new_standalone_cadence

def parse_widget(widget_info: dict) -> tuple:
    parsed_widget = None
    t = widget_info["type"]
    if t == "dropdown":
        parsed_widget = (widget_info["choices"],)
    elif t == "checkbox":
        parsed_widget = ("BOOLEAN", {"default": widget_info['default']})
    elif t == "lineedit":
        parsed_widget = ("STRING", {"default": widget_info['default']})
    elif t == "spinbox":
        parsed_widget = ("INT", {"default": widget_info['default']})
    elif t == "doublespinbox":
        parsed_widget = ("FLOAT", {"default": widget_info['default']})
    return parsed_widget


def get_node_params(input_params):
    data_info = {"required": {}, }
    if input_params:
        for name, widget_info in input_params.items():
            data_info["required"][name] = parse_widget(widget_info)
    data_info["optional"] = {"deforum_data": ("deforum_data",)}
    return data_info


class DeforumBaseParamsNode(DeforumDataBase):
    params = get_node_params(deforum_base_params)
    display_name = "Base Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumAnimParamsNode(DeforumDataBase):
    params = get_node_params(deforum_anim_params)
    display_name = "Animation Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumTranslationParamsNode(DeforumDataBase):
    params = get_node_params(deforum_translation_params)
    display_name = "Translate Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumDepthParamsNode(DeforumDataBase):
    params = get_node_params(deforum_depth_params)
    display_name = "Depth Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumNoiseParamsNode(DeforumDataBase):
    params = get_node_params(deforum_noise_params)
    display_name = "Noise Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumColorParamsNode(DeforumDataBase):
    params = get_node_params(deforum_color_coherence_params)
    display_name = "ColorMatch Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumDiffusionParamsNode(DeforumDataBase):
    params = get_node_params(deforum_diffusion_schedule_params)
    display_name = "Diffusion Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumCadenceParamsNode(DeforumDataBase):
    params = get_node_params(deforum_cadence_params)
    display_name = "Cadence Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params

class DeforumHybridParamsNode(DeforumDataBase):
    params = get_node_params(deforum_hybrid_video_params)
    display_name = "Hybrid Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params

class DeforumHybridScheduleNode(DeforumDataBase):
    params = get_node_params(deforum_hybrid_video_schedules)
    display_name = "Hybrid Schedule"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumPromptNode(DeforumDataBase):
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"forceInput": False, "multiline": True, "default": "0:'Cat Sushi'"}),
            },
            "optional": {
                "deforum_data": ("deforum_data",),
            },
        }

    RETURN_TYPES = (("deforum_data",))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum"
    display_name = "Prompt"

    @torch.inference_mode()
    def get(self, prompts, deforum_data=None):

        # Splitting the data into rows
        rows = prompts.split('\n')

        # Creating an empty dictionary
        prompts = {}

        # Parsing each row
        for row in rows:
            key, value = row.split(':', 1)
            key = int(key)
            value = value.strip('"')
            prompts[key] = value

        if deforum_data:
            deforum_data["prompts"] = prompts
        else:
            deforum_data = {"prompts": prompts}
        return (deforum_data,)


class DeforumAreaPromptNode(DeforumDataBase):

    default_area_prompt = '[{"0": [{"prompt": "a vast starscape with distant nebulae and galaxies", "x": 0, "y": 0, "w": 1024, "h": 1024, "s": 0.7}, {"prompt": "detailed sci-fi spaceship", "x": 512, "y": 512, "w": 50, "h": 50, "s": 0.7}]}, {"50": [{"prompt": "a vast starscape with distant nebulae and galaxies", "x": 0, "y": 0, "w": 1024, "h": 1024, "s": 0.7}, {"prompt": "detailed sci-fi spaceship", "x": 412, "y": 412, "w": 200, "h": 200, "s": 0.7}]}, {"100": [{"prompt": "a vast starscape with distant nebulae and galaxies", "x": 0, "y": 0, "w": 1024, "h": 1024, "s": 0.7}, {"prompt": "detailed sci-fi spaceship", "x": 112, "y": 112, "w": 800, "h": 800, "s": 0.7}]}]'
    default_prompt = "Alien landscape"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframe": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "mode":(["default", "percentage", "strength"],),
                "prompt": ("STRING", {"forceInput": False, "multiline": True, 'default': cls.default_prompt,}),
                "width": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "deforum_data": ("deforum_data",),
            },
        }

    RETURN_TYPES = (("deforum_data",))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum"
    display_name = "Area Prompt"

    @torch.inference_mode()
    def get(self, keyframe, mode, prompt, width, height, x, y, strength, deforum_data=None):

        area_prompt = {"prompt": prompt, "x": x, "y": y, "w": width, "h": height, "s": strength, "mode":mode}
        area_prompt_dict = {f"{keyframe}": [area_prompt]}

        if not deforum_data:
            deforum_data = {"area_prompts":[area_prompt_dict]}

        if "area_prompts" not in deforum_data:
            deforum_data["area_prompts"] = [area_prompt_dict]
        else:

            added = None

            for item in deforum_data["area_prompts"]:
                for k, v in item.items():
                    if int(k) == keyframe:
                        if area_prompt not in v:
                            v.append(area_prompt)
                            added = True
                        else:
                            added = True
            if not added:
                deforum_data["area_prompts"].append(area_prompt_dict)

        deforum_data["prompts"] = None

        return (deforum_data,)


class DeforumSingleSampleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "deforum_data": ("deforum_data",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",)
            },
        }

    RETURN_TYPES = (("IMAGE",))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum"
    display_name = "Integrated Pipeline"

    @torch.inference_mode()
    def get(self, deforum_data, model, clip, vae, *args, **kwargs):

        root_dict = RootArgs()
        args_dict = {key: value["value"] for key, value in DeforumArgs().items()}
        anim_args_dict = {key: value["value"] for key, value in DeforumAnimArgs().items()}
        output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}
        loop_args_dict = {key: value["value"] for key, value in LoopArgs().items()}
        parseq_args_dict = {key: value["value"] for key, value in ParseqArgs().items()}
        root = SimpleNamespace(**root_dict)
        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)
        video_args = SimpleNamespace(**output_args_dict)
        parseq_args = SimpleNamespace(**parseq_args_dict)

        parseq_args.parseq_manifest = ""

        # #parseq_args = None
        loop_args = SimpleNamespace(**loop_args_dict)
        controlnet_args = SimpleNamespace(**{"controlnet_args": "None"})

        for key, value in args.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "":
                    val = None
                else:
                    val = deforum_data[key]
                setattr(args, key, val)

        for key, value in anim_args.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = deforum_data[key]
                setattr(anim_args, key, val)

        for key, value in video_args.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = deforum_data[key]
                setattr(anim_args, key, val)

        for key, value in root.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "":
                    val = None
                else:
                    val = deforum_data[key]
                setattr(root, key, val)

        for key, value in loop_args.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "":
                    val = None
                else:
                    val = deforum_data[key]
                setattr(loop_args, key, val)

        success = None
        root.timestring = time.strftime('%Y%m%d%H%M%S')
        args.timestring = root.timestring
        args.strength = max(0.0, min(1.0, args.strength))

        root.animation_prompts = deforum_data.get("prompts", {})


        if not args.use_init and not anim_args.hybrid_use_init_image:
            args.init_image = None

        elif anim_args.animation_mode == 'Video Input':
            args.use_init = True

        current_arg_list = [args, anim_args, video_args, parseq_args, root]
        full_base_folder_path = os.path.join(os.getcwd(), "output/deforum")

        args.batch_name = f"aiNodes_Deforum_{args.timestring}"
        args.outdir = os.path.join(full_base_folder_path, args.batch_name)

        root.raw_batch_name = args.batch_name
        args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)

        # os.makedirs(args.outdir, exist_ok=True)

        def generate(*args, **kwargs):
            from .deforum_comfy_sampler import sample_deforum
            image = sample_deforum(model, clip, vae, **kwargs)

            return image

        self.deforum = DeforumAnimationPipeline(generate)

        self.deforum.config_dir = os.path.join(os.getcwd(), "output/_deforum_configs")
        os.makedirs(self.deforum.config_dir, exist_ok=True)
        # self.deforum.generate_inpaint = self.generate_inpaint
        import comfy
        pbar = comfy.utils.ProgressBar(deforum_data["max_frames"])

        def datacallback(data=None):
            if data:
                if "image" in data:
                    pbar.update_absolute(data["frame_idx"], deforum_data["max_frames"], ("JPEG", data["image"], 512))

        self.deforum.datacallback = datacallback
        deforum_data["turbo_steps"] = deforum_data.get("diffusion_cadence", 0)
        animation = self.deforum(**deforum_data)

        results = []
        for i in self.deforum.images:
            tensor = torch.from_numpy(np.array(i).astype(np.float32) / 255.0)

            results.append(tensor)
            result = torch.stack(results, dim=0)

        return (result,)


def get_current_keys(anim_args, seed, root, parseq_args=None, video_args=None, area_prompts=None):
    use_parseq = False if parseq_args == None else True
    anim_args.max_frames += 2
    keys = DeforumAnimKeys(anim_args, seed)  # if not use_parseq else ParseqAnimKeys(parseq_args, video_args)
    areas = None
    # Always enable pseudo-3d with parseq. No need for an extra toggle:
    # Whether it's used or not in practice is defined by the schedules
    if use_parseq:
        anim_args.flip_2d_perspective = True
        # expand prompts out to per-frame
    if use_parseq and keys.manages_prompts():
        prompt_series = keys.prompts
    else:
        prompt_series = None
        if hasattr(root, 'animation_prompts'):
            if root.animation_prompts is not None:
                prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames + 1)])
                for i, prompt in root.animation_prompts.items():
                    if str(i).isdigit():
                        prompt_series[int(i)] = prompt
                    else:
                        prompt_series[int(numexpr.evaluate(i))] = prompt
                prompt_series = prompt_series.ffill().bfill()
        if area_prompts is not None:
            areas = interpolate_areas(area_prompts, anim_args.max_frames)
    anim_args.max_frames -= 2
    return keys, prompt_series, areas


class DeforumCacheLatentNode:
    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "cache_index": ("INT", {"default":0, "min": 0, "max": 16, "step": 1})
            }
        }

    RETURN_TYPES = (("LATENT",))
    FUNCTION = "cache_it"
    CATEGORY = f"deforum"
    display_name = "Cache Latent"
    OUTPUT_NODE = True

    def cache_it(self, latent=None, cache_index=0):
        global deforum_cache

        if "latent" not in deforum_cache:

            deforum_cache["latent"] = {}

        deforum_cache["latent"][cache_index] = latent

        return (latent,)


class DeforumGetCachedLatentNode:

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {

            "cache_index": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1})

        }}

    RETURN_TYPES = (("LATENT",))
    FUNCTION = "get_cached_latent"
    CATEGORY = f"deforum"
    OUTPUT_NODE = True
    display_name = "Load Cached Latent"

    def get_cached_latent(self, cache_index=0):
        latent_dict = deforum_cache.get("latent", {})
        latent = latent_dict.get(cache_index)
        return (latent,)



class DeforumCacheImageNode:
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Force re-evaluation of the node
        # if autorefresh == "Yes":
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cache_index": ("INT", {"default":0, "min": 0, "max": 16, "step": 1})
            }
        }

    RETURN_TYPES = (("IMAGE",))
    FUNCTION = "cache_it"
    CATEGORY = f"deforum"
    display_name = "Cache Image"
    OUTPUT_NODE = True

    def cache_it(self, image=None, cache_index=0):
        global deforum_cache
        print("DEFORUM CACHING IMAGE ON SLOT", cache_index)
        if "image" not in deforum_cache:

            deforum_cache["image"] = {}

        deforum_cache["image"][cache_index] = image

        return (image,)


class DeforumGetCachedImageNode:

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {

            "cache_index": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1})

        }}

    RETURN_TYPES = (("IMAGE",))
    FUNCTION = "get_cached_latent"
    CATEGORY = f"deforum"
    OUTPUT_NODE = True
    display_name = "Load Cached Image"

    def get_cached_latent(self, cache_index=0):
        img_dict = deforum_cache.get("image", {})
        image = img_dict.get(cache_index)
        return (image,)


class DeforumSeedNode:
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum"
    RETURN_TYPES = (("INT",))
    display_name = "Seed Node"

    @torch.inference_mode()
    def get(self, seed, *args, **kwargs):
        return (seed,)


def get_latent_with_seed(seed):
    return torch.randn(generator=torch.manual_seed(seed))

def generate_seed_list(max_frames, mode='fixed', start_seed=0, step=1):
    """
    Generates a list of seed integers compatible with PyTorch in various manners.

    Parameters:
    - max_frames (int): The maximum number of frames/length of the seed list.
    - mode (str): The mode of seed generation, one of 'fixed', 'random', 'ladder', 'incrementing', or 'decrementing'.
    - start_seed (int): The starting seed value for modes other than 'random'.
    - step (int): The step size for 'incrementing', 'decrementing', and 'ladder' modes.

    Returns:
    - list: A list of seed integers.
    """
    if mode == 'fixed':
        return [start_seed for _ in range(max_frames)]
    elif mode == 'random':
        return [random.randint(0, 2**32 - 1) for _ in range(max_frames)]
    elif mode == 'ladder':
        # Generate a ladder sequence where the sequence is repeated after reaching the max_frames
        return [(start_seed + i // 2 * step if i % 2 == 0 else start_seed + (i // 2 + 1) * step) % (2**32) for i in range(max_frames)]
    elif mode == 'incrementing':
        return [(start_seed + i * step) % (2**32) for i in range(max_frames)]
    elif mode == 'decrementing':
        return [(start_seed - i * step) % (2**32) for i in range(max_frames)]
    else:
        raise ValueError("Invalid mode specified. Choose among 'fixed', 'random', 'ladder', 'incrementing', 'decrementing'.")




class DeforumIteratorNode:

    def __init__(self):
        self.first_run = True
        self.frame_index = 0
        self.seed = ""
        self.seeds = []

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Force re-evaluation of the node
        # if autorefresh == "Yes":
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "deforum_data": ("deforum_data",),
                "latent_type": (["stable_diffusion", "stable_cascade"],)
            },
            "optional": {
                "latent": ("LATENT",),
                "init_latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "subseed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "subseed_strength": ("FLOAT", {"default": 0.8, "min": 0, "max": 1.0}),
                "slerp_strength": ("FLOAT", {"default": 0.1, "min": 0, "max": 1.0}),
                "reset_counter":("BOOLEAN", {"default": False},),
                "reset_latent":("BOOLEAN", {"default": False},),
            }
        }

    RETURN_TYPES = (("DEFORUM_FRAME_DATA", "LATENT", "STRING", "STRING"))
    RETURN_NAMES = (("deforum_frame_data", "latent", "positive_prompt", "negative_prompt"))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum"
    display_name = "Iterator Node"


    @torch.inference_mode()
    def get(self, deforum_data, latent_type, latent=None, init_latent=None, seed=None, subseed=None, subseed_strength=None, slerp_strength=None, reset_counter=False, reset_latent=False, *args, **kwargs):
        root_dict = RootArgs()
        args_dict = {key: value["value"] for key, value in DeforumArgs().items()}
        anim_args_dict = {key: value["value"] for key, value in DeforumAnimArgs().items()}
        output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}
        loop_args_dict = {key: value["value"] for key, value in LoopArgs().items()}
        root = SimpleNamespace(**root_dict)
        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)
        anim_args.diffusion_cadence = 1
        video_args = SimpleNamespace(**output_args_dict)
        parseq_args = None
        loop_args = SimpleNamespace(**loop_args_dict)
        controlnet_args = SimpleNamespace(**{"controlnet_args": "None"})

        for key, value in args.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "":
                    val = None
                else:
                    val = deforum_data[key]
                setattr(args, key, val)

        for key, value in anim_args.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = deforum_data[key]
                setattr(anim_args, key, val)

        for key, value in video_args.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = deforum_data[key]
                setattr(anim_args, key, val)

        for key, value in root.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "":
                    val = None
                else:
                    val = deforum_data[key]
                setattr(root, key, val)

        for key, value in loop_args.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "":
                    val = None
                else:
                    val = deforum_data[key]
                setattr(loop_args, key, val)

        root.animation_prompts = deforum_data.get("prompts")

        keys, prompt_series, areas = get_current_keys(anim_args, args.seed, root, area_prompts=deforum_data.get("area_prompts"))

        if self.frame_index > anim_args.max_frames or reset_counter:
            from . import standalone_cadence
            standalone_cadence.turbo_next_image, standalone_cadence.turbo_next_frame_idx = None, 0
            standalone_cadence.turbo_prev_image, standalone_cadence.turbo_prev_frame_idx = None, 0
            self.reset_counter = False
            # self.reset_iteration()
            self.frame_index = 0
            # .should_run = False
            # return [None]
            self.first_run = True

        # else:
        args.scale = keys.cfg_scale_schedule_series[self.frame_index]
        if prompt_series is not None:
            args.prompt = prompt_series[self.frame_index]

        args.seed = int(args.seed)
        root.seed_internal = int(root.seed_internal)
        args.seed_iter_N = int(args.seed_iter_N)

        if self.seed == "":
            self.seed = args.seed
            self.seed_internal = root.seed_internal
            self.seed_iter_N = args.seed_iter_N

        self.seed = next_seed(args, root)
        args.seed = self.seed
        self.seeds.append(self.seed)

        blend_value = 0.0

        # print(frame, anim_args.diffusion_cadence, node.deforum.prompt_series)

        next_frame = self.frame_index + anim_args.diffusion_cadence
        next_prompt = None

        def generate_blend_values(distance_to_next_prompt, blend_type="linear"):
            if blend_type == "linear":
                return [i / distance_to_next_prompt for i in range(distance_to_next_prompt + 1)]
            elif blend_type == "exponential":
                base = 2
                return [1 / (1 + math.exp(-8 * (i / distance_to_next_prompt - 0.5))) for i in
                        range(distance_to_next_prompt + 1)]
            else:
                raise ValueError(f"Unknown blend type: {blend_type}")

        def find_last_prompt_change(current_index, prompt_series):
            # Step backward from the current position
            for i in range(current_index - 1, -1, -1):
                if prompt_series[i] != prompt_series[current_index]:
                    return i
            return 0  # default to the start if no change found

        def find_next_prompt_change(current_index, prompt_series):
            # Step forward from the current position
            for i in range(current_index + 1, len(prompt_series) - 1):
                if i < anim_args.max_frames:

                    if prompt_series[i] != prompt_series[current_index]:
                        return i
            return len(prompt_series) - 1  # default to the end if no change found

        if prompt_series is not None:
            last_prompt_change = find_last_prompt_change(self.frame_index, prompt_series)

            next_prompt_change = find_next_prompt_change(self.frame_index, prompt_series)

            distance_between_changes = next_prompt_change - last_prompt_change
            current_distance_from_last = self.frame_index - last_prompt_change

            # Generate blend values for the distance between prompt changes
            blend_values = generate_blend_values(distance_between_changes, blend_type="exponential")

            # Fetch the blend value based on the current frame's distance from the last prompt change
            blend_value = blend_values[current_distance_from_last]

            if len(prompt_series) - 1 > next_prompt_change:
                next_prompt = prompt_series[next_prompt_change]

        gen_args = self.get_current_frame(args, anim_args, root, keys, self.frame_index, areas)

        # self.content.frame_slider.setMaximum(anim_args.max_frames - 1)

        self.args = args
        self.root = root
        if prompt_series is not None:
            gen_args["next_prompt"] = next_prompt
            gen_args["prompt_blend"] = blend_value
        gen_args["frame_index"] = self.frame_index
        gen_args["max_frames"] = anim_args.max_frames

        seeds = generate_seed_list(anim_args.max_frames, args.seed_behavior, seed, args.seed_iter_N)
        subseeds = generate_seed_list(anim_args.max_frames, args.seed_behavior, subseed, args.seed_iter_N)

        if latent is None or reset_latent or not hasattr(self, "rng"):
            global deforum_cache
            deforum_cache.clear()
            if latent_type == "stable_diffusion":
                channels = 4
                compression = 8
            else:
                channels = 16
                compression = 42
            if init_latent is not None:
                args.height, args.width = init_latent["samples"].shape[2] * 8, init_latent["samples"].shape[3] * 8
            self.rng = ImageRNGNoise((channels, args.height // compression, args.width // compression),
                                     [seeds[self.frame_index]], [subseeds[self.frame_index]],
                                     0.6, 1024, 1024)
            if latent_type == "stable_diffusion":
                l = self.rng.first().half().to(comfy.model_management.intermediate_device())
            else:
                l = torch.zeros([1, 16, args.height // 42, args.width // 42]).to(comfy.model_management.intermediate_device())
            latent = {"samples": l}
            gen_args["denoise"] = 1.0
        else:
            if latent_type == "stable_diffusion" and slerp_strength > 0:
                args.height, args.width = latent["samples"].shape[2] * 8, latent["samples"].shape[3] * 8
                l = self.rng.next().clone().to(comfy.model_management.intermediate_device())
                s = latent["samples"].clone().to(comfy.model_management.intermediate_device())
                latent = {"samples":slerp(slerp_strength, s, l)}
        print(f"[ Deforum Iterator: {self.frame_index} / {anim_args.max_frames} {self.seed}]")
        gen_args["noise"] = self.rng
        gen_args["seed"] = int(seed)

        if self.frame_index == 0 and init_latent is not None:
            latent = init_latent
            gen_args["denoise"] = keys.strength_schedule_series[0]

        #if anim_args.diffusion_cadence > 1:
        # global turbo_prev_img, turbo_prev_frame_idx, turbo_next_image, turbo_next_frame_idx, opencv_image
        if anim_args.diffusion_cadence > 1:
            self.frame_index += anim_args.diffusion_cadence if not self.first_run else 0# if anim_args.diffusion_cadence == 1

            # if turbo_steps > 1:
            # turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            # turbo_next_image, turbo_next_frame_idx = opencv_image, self.frame_index
                # frame_idx += turbo_steps

            self.first_run = False
        else:
            self.frame_index += 1
        latent["samples"] = latent["samples"].float()
        return {"ui": {"counter":(self.frame_index,), "max_frames":(anim_args.max_frames,)}, "result": (gen_args, latent, gen_args["prompt"], gen_args["negative_prompt"],),}
        # return (gen_args, latent, gen_args["prompt"], gen_args["negative_prompt"],)

    def get_current_frame(self, args, anim_args, root, keys, frame_idx, areas=None):
        if hasattr(args, 'prompt'):
            prompt, negative_prompt = split_weighted_subprompts(args.prompt, frame_idx, anim_args.max_frames)
        else:
            prompt = ""
            negative_prompt = ""
        strength = keys.strength_schedule_series[frame_idx]

        return {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "denoise": strength,
                "cfg": args.scale,
                "steps": int(keys.steps_schedule_series[self.frame_index]),
                "root": root,
                "keys": keys,
                "frame_idx": frame_idx,
                "anim_args": anim_args,
                "args": args,
                "areas":areas[frame_idx] if areas is not None else None}




class DeforumKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "latent": ("LATENT",),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),

                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    display_name = "KSampler"
    CATEGORY = "deforum"

    def sample(self, model, latent, positive, negative, deforum_frame_data):
        from nodes import common_ksampler

        seed = deforum_frame_data.get("seed", 0)
        steps = deforum_frame_data.get("steps", 10)
        cfg = deforum_frame_data.get("cfg", 7.5)
        sampler_name = deforum_frame_data.get("sampler_name", "euler_a")
        scheduler = deforum_frame_data.get("scheduler", "normal")
        denoise = deforum_frame_data.get("denoise", 1.0)
        latent["samples"] = latent["samples"].float()
        #print("DENOISE", denoise)
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,
                               denoise=denoise)


class DeforumFrameDataExtract:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     }
                }
    RETURN_TYPES = ("INT","INT", "INT", "FLOAT", "STRING", "STRING", "FLOAT", "FLOAT")
    RETURN_NAMES = ("frame_idx", "seed", "steps", "cfg_scale", "sampler_name", "scheduler_name", "denoise", "subseed_strength")
    FUNCTION = "get_data"
    display_name = "Frame Data Extract"
    CATEGORY = "deforum"

    def get_data(self, deforum_frame_data):
        from nodes import common_ksampler
        seed = deforum_frame_data.get("seed", 0)
        steps = deforum_frame_data.get("steps", 10)
        cfg = deforum_frame_data.get("cfg", 7.5)
        sampler_name = deforum_frame_data.get("sampler_name", "euler_a")
        scheduler = deforum_frame_data.get("scheduler", "normal")
        denoise = deforum_frame_data.get("denoise", 1.0)

        keys = deforum_frame_data.get("keys")
        frame_idx = deforum_frame_data.get("frame_idx")
        subseed_str = keys.subseed_strength_schedule_series[frame_idx]

        #print("DENOISE", denoise)
        return (frame_idx, seed, steps, cfg, sampler_name, scheduler, denoise,subseed_str,)


def tensor2pil(image):
    if image is not None:
        with torch.inference_mode():
            return Image.fromarray(np.clip(255. * image.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    else:
        return None

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2np(img):
    np_img = np.array(tensor2pil(img))

    return np_img


class DeforumFrameWarpNode:
    def __init__(self):
        self.depth_model = None
        self.depth = None
        self.algo = ""
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     "warp_depth_image": ("BOOLEAN",{"default":False}),
                     }
                }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE")
    RETURN_NAMES = ("IMAGE","DEPTH", "WARPED_DEPTH")
    FUNCTION = "fn"
    display_name = "Frame Warp"
    CATEGORY = "deforum"



    def fn(self, image, deforum_frame_data, warp_depth_image):
        from deforum.models import DepthModel
        from deforum.utils.deforum_framewarp_utils import anim_frame_warp
        np_image = None
        data = deforum_frame_data
        if image is not None:
            if image.shape[0] > 1:
                for img in image:
                    np_image = tensor2np(img)
            else:
                np_image = tensor2np(image)

            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

            args = data.get("args")
            anim_args = data.get("anim_args")
            keys = data.get("keys")
            frame_idx = data.get("frame_idx")
            print(keys.translation_z_series[frame_idx])

            # print(keys.translation_z_series[frame_idx])

            if frame_idx == 0:
                self.depth = None
            predict_depths = (
                                     anim_args.animation_mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
            predict_depths = predict_depths or (
                    anim_args.hybrid_composite and anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth'])

            if self.depth_model == None or self.algo != anim_args.depth_algorithm:
                self.vram_state = "high"
                if self.depth_model is not None:
                    self.depth_model.to("cpu")
                    del self.depth_model
                    # torch_gc()

                self.algo = anim_args.depth_algorithm
                if predict_depths:
                    keep_in_vram = True if self.vram_state == 'high' else False
                    # device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else self.root.device)
                    # TODO Set device in root in webui
                    device = 'cuda'
                    self.depth_model = DepthModel("models/other", device,
                                                  keep_in_vram=keep_in_vram,
                                                  depth_algorithm=anim_args.depth_algorithm, Width=args.width,
                                                  Height=args.height,
                                                  midas_weight=anim_args.midas_weight)

                    # depth-based hybrid composite mask requires saved depth maps
                    if anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type == 'Depth':
                        anim_args.save_depth_maps = True
                else:
                    self.depth_model = None
                    anim_args.save_depth_maps = False
            if self.depth_model != None and not predict_depths:
                self.depth_model = None
            if self.depth_model is not None:
                self.depth_model.to('cuda')
            prev_depth = self.depth
            warped_np_img, self.depth, mask = anim_frame_warp(np_image, args, anim_args, keys, frame_idx,
                                                              depth_model=self.depth_model, depth=prev_depth, device='cuda',
                                                              half_precision=True)

            image = Image.fromarray(cv2.cvtColor(warped_np_img, cv2.COLOR_BGR2RGB))
            tensor = pil2tensor(image)
            if self.depth is not None:
                num_channels = len(self.depth.shape)

                if num_channels <= 3:
                    depth_image = self.depth_model.to_image(self.depth.detach().cpu())
                else:
                    depth_image = self.depth_model.to_image(self.depth[0].detach().cpu())
                ret_depth = pil2tensor(depth_image).detach().cpu()
                if warp_depth_image:
                    depth_image, _, _ = anim_frame_warp(np.array(depth_image), args, anim_args, keys, frame_idx,
                                                                      depth_model=self.depth_model, depth=prev_depth,
                                                                      device='cuda',
                                                                      half_precision=True)
                    warped_depth_image = Image.fromarray(depth_image)
                    warped_ret = pil2tensor(warped_depth_image).detach().cpu()
                else:
                    warped_ret = ret_depth

            else:
                ret_depth = tensor
                warped_ret = tensor
            # if gs.vram_state in ["low", "medium"] and self.depth_model is not None:
            #     self.depth_model.to('cpu')


            # if mask is not None:
            #     mask = mask.detach().cpu()
            #     # mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            #     mask = mask.mean(dim=0, keepdim=False)
            #     mask[mask > 1e-05] = 1
            #     mask[mask < 1e-05] = 0
            #     mask = mask[0].unsqueeze(0)

            # print(mask)
            # print(mask.shape)

            # from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.resizeRight import resizeright
            # from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.resizeRight import interp_methods
            # mask = resizeright.resize(mask, scale_factors=None,
            #                                     out_shape=[mask.shape[0], int(mask.shape[1] // 8), int(mask.shape[2] // 8)
            #                                             ],
            #                                     interp_method=interp_methods.lanczos3, support_sz=None,
            #                                     antialiasing=True, by_convs=True, scale_tolerance=None,
            #                                     max_numerator=10, pad_mode='reflect')
            # print(mask.shape)
            return (tensor, ret_depth,warped_ret,)
            # return [data, tensor, mask, ret_depth, self.depth_model]
        else:
            return (image, image,image,)


class DeforumColorMatchNode:

    def __init__(self):
        self.depth_model = None
        self.algo = ""
        self.color_match_sample = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     "force_use_sample": ("BOOLEAN", {"default":False},)
                     },
                "optional":
                    {"force_sample_image":("IMAGE",)}
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fn"
    display_name = "Color Match"
    CATEGORY = "deforum"



    def fn(self, image, deforum_frame_data, force_use_sample, force_sample_image=None):
        if image is not None:
            anim_args = deforum_frame_data.get("anim_args")
            image = np.array(tensor2pil(image))
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frame_idx = deforum_frame_data.get("frame_idx", 0)
            if frame_idx == 0 and not force_use_sample:
                self.color_match_sample = image.copy()
                return (pil2tensor(image),)
            if force_use_sample:
                if force_sample_image is not None:
                    self.color_match_sample = np.array(tensor2pil(force_sample_image)).copy()
            if anim_args.color_coherence != 'None' and self.color_match_sample is not None:
                image = maintain_colors(image, self.color_match_sample, anim_args.color_coherence)
            print(f"[ Deforum Color Coherence: {anim_args.color_coherence} ]")
            if self.color_match_sample is None:
                self.color_match_sample = image.copy()
            if anim_args.color_force_grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = pil2tensor(image)

        return (image,)


class DeforumAddNoiseNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fn"
    display_name = "Add Noise"
    CATEGORY = "deforum"

    def fn(self, image, deforum_frame_data):

        if image is not None:
            keys = deforum_frame_data.get("keys")
            args = deforum_frame_data.get("args")
            anim_args = deforum_frame_data.get("anim_args")
            root = deforum_frame_data.get("root")
            frame_idx = deforum_frame_data.get("frame_idx")
            noise = keys.noise_schedule_series[frame_idx]
            kernel = int(keys.kernel_schedule_series[frame_idx])
            sigma = keys.sigma_schedule_series[frame_idx]
            amount = keys.amount_schedule_series[frame_idx]
            threshold = keys.threshold_schedule_series[frame_idx]
            contrast = keys.contrast_schedule_series[frame_idx]
            if anim_args.use_noise_mask and keys.noise_mask_schedule_series[frame_idx] is not None:
                noise_mask_seq = keys.noise_mask_schedule_series[frame_idx]
            else:
                noise_mask_seq = None
            mask_vals = {}
            noise_mask_vals = {}

            mask_vals['everywhere'] = Image.new('1', (args.width, args.height), 1)
            noise_mask_vals['everywhere'] = Image.new('1', (args.width, args.height), 1)

            # from ainodes_frontend.nodes.deforum_nodes.deforum_framewarp_node import tensor2np
            prev_img = tensor2np(image)
            mask_image = None
            # apply scaling
            contrast_image = (prev_img * contrast).round().astype(np.uint8)
            # anti-blur
            if amount > 0:
                contrast_image = unsharp_mask(contrast_image, (kernel, kernel), sigma, amount, threshold,
                                              mask_image if args.use_mask else None)
            # apply frame noising
            if args.use_mask or anim_args.use_noise_mask:
                root.noise_mask = compose_mask_with_check(root, args, noise_mask_seq,
                                                          noise_mask_vals,
                                                          Image.fromarray(
                                                              cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)))
            noised_image = add_noise(contrast_image, noise, int(args.seed), anim_args.noise_type,
                                     (anim_args.perlin_w, anim_args.perlin_h,
                                      anim_args.perlin_octaves,
                                      anim_args.perlin_persistence),
                                     root.noise_mask, args.invert_mask)
            # image = Image.fromarray(noised_image)
            print(f"[ Deforum Adding Noise: {noise} {anim_args.noise_type}]")
            image = pil2tensor(noised_image).detach().cpu()

            return (image,)


class DeforumHybridMotionNode:
    raft_model = None
    methods = ['RAFT', 'DIS Medium', 'DIS Fine', 'Farneback']

    def __init__(self):
        self.prev_image = None
        self.flow = None
        self.image_size = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "hybrid_image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     "hybrid_method": ([s.methods]),
                     }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fn"
    display_name = "Hybrid Motion"
    CATEGORY = "deforum"

    def fn(self, image, hybrid_image, deforum_frame_data, hybrid_method):
        if self.raft_model is None:
            self.raft_model = RAFT()

        flow_factor = deforum_frame_data["keys"].hybrid_flow_factor_schedule_series[deforum_frame_data["frame_index"]]
        p_img = tensor2pil(image)
        size = p_img.size

        pil_image = np.array(p_img).astype(np.uint8)

        if self.image_size != size:
            self.prev_image = None
            self.flow = None
            self.image_size = size

        bgr_image = cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)

        if hybrid_image is None:
            if self.prev_image is None:
                self.prev_image = bgr_image
                return (image,)
            else:
                self.flow = get_flow_from_images(self.prev_image, bgr_image, hybrid_method, self.raft_model, self.flow)

                self.prev_image = copy.deepcopy(bgr_image)

                bgr_image = image_transform_optical_flow(bgr_image, self.flow, flow_factor)

                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                return (pil2tensor(rgb_image),)
        else:

            pil_image_ref = np.array(tensor2pil(hybrid_image).resize((self.image_size), Image.Resampling.LANCZOS)).astype(np.uint8)
            bgr_image_ref = cv2.cvtColor(pil_image_ref, cv2.COLOR_RGB2BGR)
            bgr_image_ref = cv2.resize(bgr_image_ref, (bgr_image.shape[1], bgr_image.shape[0]))
            if self.prev_image is None:
                self.prev_image = bgr_image_ref
                return (image,)
            else:
                self.flow = get_flow_from_images(self.prev_image, bgr_image_ref, hybrid_method, self.raft_model,
                                                 self.flow)
                bgr_image = image_transform_optical_flow(bgr_image, self.flow, flow_factor)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                return (pil2tensor(rgb_image),)

class DeforumSetVAEDownscaleRatioNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"vae": ("VAE",),
                     "downscale_ratio": ("INT", {"default": 42, "min": 32, "max": 64, "step": 1}),
                     },
                }
    RETURN_TYPES = ("VAE",)
    FUNCTION = "fn"
    display_name = "Set VAE Downscale Ratio"
    CATEGORY = "deforum"

    def fn(self, vae, downscale_ratio):
        vae.downscale_ratio = downscale_ratio
        return (vae,)


class DeforumLoadVideo:

    def __init__(self):
        self.video_path = None

    # @classmethod
    # def INPUT_TYPES(cls):
    #     input_dir = folder_paths.get_input_directory()
    #     files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    #     video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov'))]  # Add more video formats as needed
    #     return {"required":
    #                 {"video": (sorted(video_files), {"file_upload": True})},
    #             }
    # @classmethod
    # def INPUT_TYPES(s):
    #     input_dir = folder_paths.get_input_directory()
    #     files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    #     return {"required":
    #                 {"image": (sorted(files), {"image_upload": True})},
    #             }
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in video_extensions):
                    files.append(f)
        return {"required": {
                    "video": (sorted(files),),},}

    CATEGORY = "deforum"
    display_name = "Load Video"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_video_frame"

    def __init__(self):
        self.cap = None
        self.current_frame = None

    def load_video_frame(self, video):
        video_path = folder_paths.get_annotated_filepath(video)

        # Initialize or reset video capture
        if self.cap is None or self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or self.video_path != video_path:
            try:
                self.cap.release()
            except:
                pass
            self.cap = cv2.VideoCapture(video_path)

            self.cap = cv2.VideoCapture(video_path)
            self.current_frame = -1
            self.video_path = video_path



        success, frame = self.cap.read()
        if success:
            self.current_frame += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame).astype(np.float32)
            frame = pil2tensor(frame)  # Convert to torch tensor
        else:
            # Reset if reached the end of the video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.cap.read()
            self.current_frame = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame).astype(np.float32)
            frame = pil2tensor(frame)  # Convert to torch tensor

        return (frame,)

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True

def find_next_index(output_dir, filename_prefix):
    """
    Finds the next index for an MP4 file given an output directory and a filename prefix.

    Parameters:
    - output_dir: The directory where the MP4 files are saved.
    - filename_prefix: The prefix for the filenames.

    Returns:
    - An integer representing the next index for a new MP4 file.
    """
    # Compile a regular expression pattern to match the filenames
    # This assumes the index is at the end of the filename, before the .mp4 extension
    pattern = re.compile(rf"^{re.escape(filename_prefix)}_(\d+)\.mp4$")

    max_index = -1
    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            # Extract the current index from the filename
            current_index = int(match.group(1))
            # Update the max index found
            if current_index > max_index:
                max_index = current_index

    # The next index is one more than the highest index found
    next_index = max_index + 1
    return next_index


import base64
from io import BytesIO
def tensor2pil(image):
    if image is not None:
        with torch.inference_mode():
            return Image.fromarray(np.clip(255. * image.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    else:
        return None

def pil_image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="WEBP")  # Or JPEG
    return base64.b64encode(buffer.getvalue()).decode()

class DeforumVideoSaveNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.images = []
        self.size = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "filename_prefix": ("STRING",{"default":"deforum_"}),
                     "fps": ("INT", {"default": 24, "min": 1, "max": 10000},),
                     "dump_by": (["max_frames", "per_N_frames"],),
                     "dump_every": ("INT", {"default": 0, "min": 0, "max": 4096},),
                     "dump_now": ("BOOLEAN", {"default": False},),
                     },
                "optional":
                    {"deforum_frame_data": ("DEFORUM_FRAME_DATA",),}

        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True

    FUNCTION = "fn"
    display_name = "Save Video"
    CATEGORY = "deforum"
    def add_image(self, image):
        pil_image = tensor2pil(image.unsqueeze(0))
        size = pil_image.size
        if size != self.size:
            self.size = size
            self.images.clear()
        self.images.append(np.array(pil_image).astype(np.uint8))

    def fn(self, image, filename_prefix, fps, dump_by, dump_every, dump_now, deforum_frame_data={}):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir)

        counter = find_next_index(full_output_folder, filename_prefix)

        #frame_idx = deforum_frame_data["frame_idx"]

        anim_args = deforum_frame_data.get("anim_args")
        if anim_args is not None:
            max_frames = anim_args.max_frames
        else:
            max_frames = image.shape[0] + len(self.images) + 1

        if image.shape[0] > 1:
            for img in image:
                self.add_image(img)
        else:
            self.add_image(image[0])

        print(f"[DEFORUM VIDEO SAVE NODE] holding {len(self.images)} images")
        # When the current frame index reaches the last frame, save the video

        if dump_by == "max_frames":
            dump = len(self.images) >= max_frames
        else:
            dump = len(self.images) >= dump_every

        if dump or dump_now:  # frame_idx is 0-based
            if len(self.images) >= 2:
                output_path = os.path.join(full_output_folder, f"{filename}_{counter}.mp4")
                writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=10, pixelformat='yuv420p')

                for frame in tqdm(self.images, desc="Saving MP4 (imageio)"):
                    writer.append_data(frame)
                writer.close()
                self.images = []  # Empty the list for next use
        return {"ui": {"counter":(len(self.images),), "should_dump":(dump or dump_now,), "frames":([pil_image_to_base64(tensor2pil(i)) for i in image]), "fps":(fps,)}, "result": (image,)}
    @classmethod
    def IS_CHANGED(s, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

class DeforumFILMInterpolationNode:
    def __init__(self):
        self.FILM_temp = []
        self.model = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "inter_amount": ("INT", {"default": 2, "min": 1, "max": 10000},),
                     "skip_first": ("BOOLEAN", {"default":True}),
                     "skip_last": ("BOOLEAN", {"default":False}),

                     }
                }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("POSITIVE", "NEGATIVE")
    FUNCTION = "fn"
    display_name = "FILM Interpolation"
    CATEGORY = "deforum"
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        # Force re-evaluation of the node
        return float("NaN")

    def interpolate(self, image, inter_frames, skip_first, skip_last):

        if self.model is None:
            self.model = FilmModel()
            self.model.model.cuda()

        return_frames = []
        pil_image = tensor2pil(image.clone().detach())
        np_image = np.array(pil_image.convert("RGB"))
        self.FILM_temp.append(np_image)
        if len(self.FILM_temp) == 2:

            # with torch.inference_mode():
            with torch.no_grad():
                frames = self.model.inference(self.FILM_temp[0], self.FILM_temp[1], inter_frames=inter_frames)
            # skip_first, skip_last = True, False
            if skip_first:
                frames.pop(0)
            if skip_last:
                frames.pop(-1)

            for frame in frames:
                tensor = pil2tensor(frame)[0]
                return_frames.append(tensor.detach().cpu())
            self.FILM_temp = [self.FILM_temp[1]]
        print(f"[ FILM NODE: Created {len(return_frames)} frames ]")
        if len(return_frames) > 0:
            return_frames = torch.stack(return_frames, dim=0)
            return return_frames
        else:
            return image.unsqueeze(0)


    def fn(self, image, inter_amount, skip_first, skip_last):
        result = []

        if image.shape[0] > 1:
            for img in image:
                interpolated_frames = self.interpolate(img, inter_amount, skip_first, skip_last)

                for f in interpolated_frames:
                    result.append(f)

            ret = torch.stack(result, dim=0)
        else:
            ret = self.interpolate(image[0], inter_amount, skip_first, skip_last)
        return (ret,)

class DeforumSimpleInterpolationNode:
    def __init__(self):
        self.FILM_temp = []
        self.model = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "method": (["DIS Medium", "DIS Fast", "DIS UltraFast", "Farneback Fine", "Normal"],), # "DenseRLOF", "SF",
                     "inter_amount": ("INT", {"default": 2, "min": 1, "max": 10000},),
                     "skip_first": ("BOOLEAN", {"default":False}),
                     "skip_last": ("BOOLEAN", {"default":False}),
                     }
                }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGES", "LAST_IMAGE")
    FUNCTION = "fn"
    display_name = "Simple Interpolation"
    CATEGORY = "deforum"
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        # Force re-evaluation of the node
        return float("NaN")

    def interpolate(self, image, method, inter_frames, skip_first, skip_last):

        return_frames = []
        pil_image = tensor2pil(image.clone().detach())
        np_image = np.array(pil_image.convert("RGB"))
        self.FILM_temp.append(np_image)
        if len(self.FILM_temp) == 2:

            # with torch.inference_mode():
            # frames = self.model.inference(self.FILM_temp[0], self.FILM_temp[1], inter_frames=inter_frames)

            if inter_frames > 1:
                from .interp import optical_flow_cadence

                frames = optical_flow_cadence(self.FILM_temp[0], self.FILM_temp[1], inter_frames + 1, method)
                # skip_first, skip_last = True, False
                if skip_first:
                    frames.pop(0)
                if skip_last:
                    frames.pop(-1)

                for frame in frames:
                    tensor = pil2tensor(frame)[0]
                    return_frames.append(tensor)
            else:
                return_frames = [i for i in pil2tensor(self.FILM_temp)[0]]
            self.FILM_temp = [self.FILM_temp[1]]
        print(f"[ Simple Interpolation Node: Created {len(return_frames)} frames ]")
        if len(return_frames) > 0:
            return_frames = torch.stack(return_frames, dim=0)
            return return_frames
        else:
            return image.unsqueeze(0)


    def fn(self, image, method, inter_amount, skip_first, skip_last):
        result = []

        if image.shape[0] > 1:
            for img in image:
                interpolated_frames = self.interpolate(img, method, inter_amount, skip_first, skip_last)

                for f in interpolated_frames:
                    result.append(f)

            ret = torch.stack(result, dim=0)
        else:
            ret = self.interpolate(image[0], method, inter_amount, skip_first, skip_last)

        return (ret, ret[-1].unsqueeze(0),)


class DeforumCadenceNode:
    def __init__(self):
        self.FILM_temp = []
        self.model = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     }
                }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    # RETURN_NAMES = ("POSITIVE", "NEGATIVE")
    FUNCTION = "fn"
    display_name = "Cadence Interpolation"
    CATEGORY = "deforum"

    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        # Force re-evaluation of the node
        return float("NaN")

    def interpolate(self, image, deforum_frame_data):
        global deforum_depth_algo
        #global turbo_prev_image, turbo_prev_frame_idx, turbo_next_image, turbo_next_frame_idx, opencv_image
        from . import standalone_cadence# import turbo_prev_image, turbo_next_image, turbo_next_frame_idx, turbo_prev_frame_idx
        return_frames = []
        pil_image = tensor2pil(image.clone().detach())
        np_image = np.array(pil_image.convert("RGB"))
        # self.FILM_temp.append(np_image)
        args = deforum_frame_data["args"]
        anim_args = deforum_frame_data["anim_args"]
        predict_depths = (
                                 anim_args.animation_mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
        predict_depths = predict_depths or (
                anim_args.hybrid_composite and anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth'])

        if "depth_model" not in deforum_models or deforum_depth_algo != anim_args.depth_algorithm:
            self.vram_state = "high"
            if "depth_model" in deforum_models:
                deforum_models["depth_model"].to("cpu")
                del deforum_models["depth_model"]
                # torch_gc()

            deforum_depth_algo = anim_args.depth_algorithm
            if predict_depths:
                keep_in_vram = True if self.vram_state == 'high' else False
                # device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else self.root.device)
                # TODO Set device in root in webui
                device = 'cuda'
                deforum_models["depth_model"] = DepthModel("models/other", device,
                                              keep_in_vram=keep_in_vram,
                                              depth_algorithm=anim_args.depth_algorithm, Width=args.width,
                                              Height=args.height,
                                              midas_weight=anim_args.midas_weight)

                # depth-based hybrid composite mask requires saved depth maps
                if anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type == 'Depth':
                    anim_args.save_depth_maps = True
            else:
                deforum_models["depth_model"] = None
                anim_args.save_depth_maps = False
        if deforum_models["depth_model"] != None and not predict_depths:
            deforum_models["depth_model"] = None
        if deforum_models["depth_model"] is not None:
            deforum_models["depth_model"].to('cuda')
        if "raft_model" not in deforum_models:
            deforum_models["raft_model"] = RAFT()

        # if len(self.FILM_temp) == 2:
        #global turbo_prev_image, turbo_prev_frame_idx, turbo_next_image, turbo_next_frame_idx, opencv_image
        if deforum_frame_data["frame_idx"] == 0:
            deforum_frame_data["frame_idx"] += anim_args.diffusion_cadence
        standalone_cadence.turbo_prev_image, standalone_cadence.turbo_prev_frame_idx = standalone_cadence.turbo_next_image, standalone_cadence.turbo_next_frame_idx
        standalone_cadence.turbo_next_image, standalone_cadence.turbo_next_frame_idx = np_image, deforum_frame_data["frame_idx"]

        # with torch.inference_mode():
        with torch.no_grad():

            # frames, _, _ = standalone_cadence(self.FILM_temp[0],
            #                             self.FILM_temp[1],
            #                             deforum_frame_data["frame_idx"],
            #                             anim_args.diffusion_cadence,
            #                             deforum_frame_data["args"],
            #                             deforum_frame_data["anim_args"],
            #                             deforum_frame_data["keys"],
            #                             deforum_models["raft_model"],
            #                             deforum_models["depth_model"]
            #                             )
            frames = new_standalone_cadence(deforum_frame_data["args"],
                                            deforum_frame_data["anim_args"],
                                            deforum_frame_data["root"],
                                            deforum_frame_data["keys"],
                                            deforum_frame_data["frame_idx"],
                                            deforum_models["depth_model"],
                                            deforum_models["raft_model"])


        # skip_first, skip_last = True, False
        # x = 0
        for frame in frames:
            # img = Image.fromarray(frame)
            # img.save(f"cadence_{x}.png", "PNG")
            # x += 1
            tensor = pil2tensor(frame)


            return_frames.append(tensor.squeeze(0))
            #self.FILM_temp = [self.FILM_temp[1]]
        print(f"[ FILM NODE: Created {len(return_frames)} frames ]")
        if len(return_frames) > 0:
            return_frames = torch.stack(return_frames, dim=0)
            return return_frames
        else:
            return image

    def fn(self, image, deforum_frame_data):
        result = []

        # Check if there are multiple images in the batch
        if image.shape[0] > 1:
            for img in image:
                # Ensure img has batch dimension of 1 for interpolation
                interpolated_frames = self.interpolate(img.unsqueeze(0), deforum_frame_data)

                # Collect all interpolated frames
                for f in interpolated_frames:
                    result.append(f)
            # Stack all results into a single tensor, preserving color channels
            ret = torch.stack(result, dim=0)
        else:
            # Directly interpolate if only one image is present
            ret = self.interpolate(image, deforum_frame_data)

        # The issue might be with how the last frame is being extracted or processed.
        # Ensure the last frame has the correct shape and color channels.
        # Adding unsqueeze(0) to keep the batch dimension consistent.
        last = ret[-1].unsqueeze(0)  # Preserve the last frame separately with batch dimension

        # Ensure the color information is consistent (RGB channels)
        if last.shape[1] != 3:
            # This is just a placeholder check; you might need a different check or fix based on your specific context.
            print("Warning: The last frame does not have 3 color channels. Check your interpolate function.")

        return (ret, last,)



def pyramid_blend(tensor1, tensor2, blend_value):
    # For simplicity, we'll use two levels of blending
    downsampled1 = F.avg_pool2d(tensor1, 2)
    downsampled2 = F.avg_pool2d(tensor2, 2)

    blended_low = (1 - blend_value) * downsampled1 + blend_value * downsampled2
    blended_high = tensor1 + tensor2 - F.interpolate(blended_low, scale_factor=2)

    return blended_high
def gaussian_blend(tensor2, tensor1, blend_value):
    sigma = 0.5  # Adjust for desired smoothness
    weight = math.exp(-((blend_value - 0.5) ** 2) / (2 * sigma ** 2))
    return (1 - weight) * tensor1 + weight * tensor2
def sigmoidal_blend(tensor1, tensor2, blend_value):
    # Convert blend_value into a tensor with the same shape as tensor1 and tensor2
    blend_tensor = torch.full_like(tensor1, blend_value)
    weight = 1 / (1 + torch.exp(-10 * (blend_tensor - 0.5)))  # Sigmoid function centered at 0.5
    return (1 - weight) * tensor1 + weight * tensor2


blend_methods = ["linear", "sigmoidal", "gaussian", "pyramid", "none"]

def blend_tensors(obj1, obj2, blend_value, blend_method="linear"):
    """
    Blends tensors in two given objects based on a blend value using various blending strategies.
    """

    if blend_method == "linear":
        weight = blend_value
        blended_cond = (1 - weight) * obj1[0] + weight * obj2[0]
        blended_pooled = (1 - weight) * obj1[1]['pooled_output'] + weight * obj2[1]['pooled_output']

    elif blend_method == "sigmoidal":
        blended_cond = sigmoidal_blend(obj1[0], obj2[0], blend_value)
        blended_pooled = sigmoidal_blend(obj1[1]['pooled_output'], obj2[1]['pooled_output'], blend_value)

    elif blend_method == "gaussian":
        blended_cond = gaussian_blend(obj1[0], obj2[0], blend_value)
        blended_pooled = gaussian_blend(obj1[1]['pooled_output'], obj2[1]['pooled_output'], blend_value)

    elif blend_method == "pyramid":
        blended_cond = pyramid_blend(obj1[0], obj2[0], blend_value)
        blended_pooled = pyramid_blend(obj1[1]['pooled_output'], obj2[1]['pooled_output'], blend_value)

    return [[blended_cond, {"pooled_output": blended_pooled}]]

class DeforumConditioningBlendNode:
    def __init__(self):
        self.prompt = None
        self.n_prompt = None
        self.cond = None
        self.n_cond = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"clip": ("CLIP",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     "blend_method": ([blend_methods]),
                     }
                }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("POSITIVE", "NEGATIVE")
    FUNCTION = "fn"
    display_name = "Blend Conditionings"
    CATEGORY = "deforum"
    def fn(self, clip, deforum_frame_data, blend_method):
        areas = deforum_frame_data.get("areas")
        negative_prompt = deforum_frame_data.get("negative_prompt", "")
        n_cond = self.get_conditioning(prompt=negative_prompt, clip=clip)

        if not areas:
            prompt = deforum_frame_data.get("prompt", "")
            next_prompt = deforum_frame_data.get("next_prompt", None)
            print(f"[ Deforum Conds: {prompt}, {negative_prompt} ]")
            cond = self.get_conditioning(prompt=prompt, clip=clip)
            # image = self.getInputData(2)
            # controlnet = self.getInputData(3)

            prompt_blend = deforum_frame_data.get("prompt_blend", 0.0)
            #method = self.content.blend_method.currentText()
            if blend_method != 'none':
                if next_prompt != prompt and prompt_blend != 0.0 and next_prompt is not None:
                    next_cond = self.get_conditioning(prompt=next_prompt, clip=clip)
                    cond = blend_tensors(cond[0], next_cond[0], prompt_blend, blend_method)
                    print(f"[ Deforum Cond Blend: {next_prompt}, {prompt_blend} ]")
        else:
            from nodes import ConditioningSetArea
            area_setter = ConditioningSetArea()
            cond = []
            for area in areas:
                prompt = area.get("prompt", None)
                if prompt:

                    new_cond = self.get_conditioning(clip=clip, prompt=area["prompt"])
                    new_cond = area_setter.append(conditioning=new_cond, width=int(area["w"]), height=int(area["h"]), x=int(area["x"]),
                                                  y=int(area["y"]), strength=area["s"])[0]
                    cond += new_cond

        return (cond, n_cond,)

    def get_conditioning(self, prompt="", clip=None, progress_callback=None):


        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

# class DeforumSD1_1_Node:
#
#     def __init__(self):
#         self.pipe = None
#
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required":
#                     {"prompt": ("STRING", {"forceInput": False, "multiline": True, "default": "0:'Cat Sushi'"}),
#                      "steps": ("INT", {"default": 0, "min": 0, "max": 1000}),
#                      "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0, "max": 25.0}),
#                      "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
#                      }
#                 }
#
#     RETURN_TYPES = ("IMAGE",)
#     FUNCTION = "generate"
#     display_name = "1.1 Node"
#     CATEGORY = "deforum"
#
#     def generate(self, prompt, steps, guidance_scale, seed):
#         if not self.pipe:
#             from diffusers import StableDiffusionPipeline
#             model_id = "CompVis/stable-diffusion-v1-1"
#             device = "cuda"
#             self.pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
#         image = self.pipe(prompt,
#                           num_inference_steps=steps,
#                           guidance_scale=guidance_scale,
#                           generator=torch.manual_seed(seed)).images[0]
#         t = pil2tensor(image)
#         return (t,)

schedule_types = [
    "angle", "transform_center_x", "transform_center_y", "zoom", "translation_x", "translation_y", "translation_z",
    "rotation_3d_x", "rotation_3d_y", "rotation_3d_z", "perspective_flip_theta", "perspective_flip_phi",
    "perspective_flip_gamma", "perspective_flip_fv", "noise_schedule", "strength_schedule", "contrast_schedule",
    "cfg_scale_schedule", "ddim_eta_schedule", "ancestral_eta_schedule", "pix2pix_img_cfg_scale", "subseed_schedule",
    "subseed_strength_schedule", "checkpoint_schedule", "steps_schedule", "seed_schedule", "sampler_schedule",
    "clipskip_schedule", "noise_multiplier_schedule", "mask_schedule", "noise_mask_schedule", "kernel_schedule",
    "sigma_schedule", "amount_schedule", "threshold_schedule", "aspect_ratio", "fov", "near",
    "cadence_flow_factor_schedule", "redo_flow_factor_schedule", "far", "hybrid_comp_alpha_schedule",
    "hybrid_comp_mask_blend_alpha_schedule", "hybrid_comp_mask_contrast_schedule",
    "hybrid_comp_mask_auto_contrast_cutoff_high_schedule", "hybrid_comp_mask_auto_contrast_cutoff_low_schedule",
    "hybrid_flow_factor_schedule"
]



class DeforumAmplitudeToKeyframeSeriesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"type_name": (schedule_types,),
                     "amplitude": ("AMPLITUDE",),
                     },
                "optional":
                    {
                        "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                        "math": ("STRING", {"default":"/1000"})
                    }
                }

    RETURN_TYPES = ("DEFORUM_FRAME_DATA", "AMPLITUDE")
    # RETURN_NAMES = ("POSITIVE", "NEGATIVE")
    FUNCTION = "convert"
    display_name = "Amplitude to Schedule"
    CATEGORY = "deforum"
    def safe_eval(self, expr, t, x):
        # Allowed functions and variables
        allowed_locals = {
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "log": math.log,
            "t": t,  # Current frame index
            "x": x,  # Current amplitude value
        }

        # Evaluate the expression safely
        try:
            return eval(expr, {"__builtins__": {}}, allowed_locals)
        except NameError as e:
            raise ValueError(f"Invalid expression: {e}")
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {e}")

    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        # Force re-evaluation of the node
        return float("NaN")

    def convert(self, type_name, amplitude, deforum_frame_data={}, math="x/100"):


        # Apply the math expression to each element of the amplitude series
        frame_index = deforum_frame_data.get("frame_idx", 0)
        modified_amplitude_list = []

        # Apply the math expression to each element of the amplitude list
        for x in amplitude:
            modified_value = self.safe_eval(math, frame_index, x)
            modified_amplitude_list.append(modified_value)
        modified_amplitude_series = pd.Series(modified_amplitude_list)

        if "keys" in deforum_frame_data:
            setattr(deforum_frame_data["keys"], f"{type_name}_series", modified_amplitude_series)
        return (deforum_frame_data, modified_amplitude_list,)


class DeforumAmplitudeToString:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"amplitude": ("AMPLITUDE",),
                     },
                }

    RETURN_TYPES = ("STRING",)
    # RETURN_NAMES = ("POSITIVE", "NEGATIVE")
    FUNCTION = "convert"
    display_name = "Amplitude to String"
    CATEGORY = "deforum"


    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        # Force re-evaluation of the node
        return float("NaN")

    def convert(self, amplitude):

        return (str(amplitude),)


class DeforumControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
                             },
                "optional": {"image": ("IMAGE",)}
                }
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet"
    display_name = "Apply ControlNet [safe]"

    CATEGORY = "deforum"

    def apply_controlnet(self, conditioning, control_net, strength, image=None):
        if strength == 0 or image is None:
            return (conditioning, )
        c = []
        control_hint = image.movedim(-1,1)
        for t in conditioning:
            n = [t[0], t[1].copy()]
            c_net = control_net.copy().set_cond_hint(control_hint, strength)
            if 'control' in t[1]:
                c_net.set_previous_controlnet(t[1]['control'])
            n[1]['control'] = c_net
            n[1]['control_apply_to_uncond'] = True
            c.append(n)
        return (c, )





# Create an empty dictionary for class mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Iterate through all classes defined in your code
# Import the deforum_nodes.deforum_node module
deforum_node_module = importlib.import_module('deforum-comfy-nodes.deforum_nodes.deforum_node')

# Iterate through all classes defined in deforum_nodes.deforum_node
for name, obj in inspect.getmembers(deforum_node_module):
    # Check if the member is a class
    if inspect.isclass(obj) and hasattr(obj, "INPUT_TYPES"):
        # Extract the class name and display name
        class_name = name
        display_name = getattr(obj, "display_name", name)  # Use class attribute or default to class name
        # Add the class to the mappings
        NODE_CLASS_MAPPINGS[class_name] = obj
        NODE_DISPLAY_NAME_MAPPINGS[name] = "(deforum) " + display_name
