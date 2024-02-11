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

from deforum import DeforumAnimationPipeline, ImageRNGNoise
from deforum.generators.deforum_flow_generator import get_flow_from_images
from deforum.generators.deforum_noise_generator import add_noise
from deforum.models import RAFT
from deforum.pipeline_utils import next_seed
from deforum.pipelines.deforum_animation.animation_helpers import DeforumAnimKeys
from deforum.pipelines.deforum_animation.animation_params import RootArgs, DeforumArgs, DeforumAnimArgs, \
    DeforumOutputArgs, LoopArgs, ParseqArgs
from deforum.utils.image_utils import maintain_colors, unsharp_mask, compose_mask_with_check, \
    image_transform_optical_flow
from deforum.utils.string_utils import substitute_placeholders, split_weighted_subprompts
from .deforum_ui_data import (deforum_base_params, deforum_anim_params, deforum_translation_params,
                              deforum_cadence_params, deforum_masking_params, deforum_depth_params,
                              deforum_noise_params, deforum_color_coherence_params, deforum_diffusion_schedule_params,
                              deforum_hybrid_video_params, deforum_video_init_params, deforum_image_init_params,
                              deforum_hybrid_video_schedules)
from .deforum_node_base import DeforumDataBase
import torch.nn.functional as F

deforum_cache = {}


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
    display_name = "Deforum Base Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumAnimParamsNode(DeforumDataBase):
    params = get_node_params(deforum_anim_params)
    display_name = "Deforum Animation Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumTranslationParamsNode(DeforumDataBase):
    params = get_node_params(deforum_translation_params)
    display_name = "Deforum Translate Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumDepthParamsNode(DeforumDataBase):
    params = get_node_params(deforum_depth_params)
    display_name = "Deforum Depth Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumNoiseParamsNode(DeforumDataBase):
    params = get_node_params(deforum_noise_params)
    display_name = "Deforum Noise Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumColorParamsNode(DeforumDataBase):
    params = get_node_params(deforum_color_coherence_params)
    display_name = "Deforum ColorMatch Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumDiffusionParamsNode(DeforumDataBase):
    params = get_node_params(deforum_diffusion_schedule_params)
    display_name = "Deforum Diffusion Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params


class DeforumCadenceParamsNode(DeforumDataBase):
    params = get_node_params(deforum_cadence_params)
    display_name = "Deforum Cadence Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params

class DeforumHybridParamsNode(DeforumDataBase):
    params = get_node_params(deforum_hybrid_video_params)
    display_name = "Deforum Hybrid Parameters"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(s):
        return s.params

class DeforumHybridScheduleNode(DeforumDataBase):
    params = get_node_params(deforum_hybrid_video_schedules)
    display_name = "Deforum Hybrid Schedule"

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
    CATEGORY = f"deforum_data"
    display_name = "Deforum Prompt"

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


# class DeforumSampleNode:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "deforum_data": ("deforum_data",),
#                 "model": ("MODEL",),
#                 "clip": ("CLIP",),
#                 "vae": ("VAE",)
#             },
#         }
#
#     RETURN_TYPES = (("IMAGE",))
#     FUNCTION = "get"
#     OUTPUT_NODE = True
#     CATEGORY = f"deforum_sampling"
#     display_name = "Deforum KSampler"
#
#     @torch.inference_mode()
#     def get(self, deforum_data, model, clip, vae, *args, **kwargs):
#
#         root_dict = RootArgs()
#         args_dict = {key: value["value"] for key, value in DeforumArgs().items()}
#         anim_args_dict = {key: value["value"] for key, value in DeforumAnimArgs().items()}
#         output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}
#         loop_args_dict = {key: value["value"] for key, value in LoopArgs().items()}
#         parseq_args_dict = {key: value["value"] for key, value in ParseqArgs().items()}
#         root = SimpleNamespace(**root_dict)
#         args = SimpleNamespace(**args_dict)
#         anim_args = SimpleNamespace(**anim_args_dict)
#         video_args = SimpleNamespace(**output_args_dict)
#         parseq_args = SimpleNamespace(**parseq_args_dict)
#
#         parseq_args.parseq_manifest = ""
#
#         # #parseq_args = None
#         loop_args = SimpleNamespace(**loop_args_dict)
#         controlnet_args = SimpleNamespace(**{"controlnet_args": "None"})
#
#         for key, value in args.__dict__.items():
#             if key in deforum_data:
#                 if deforum_data[key] == "":
#                     val = None
#                 else:
#                     val = deforum_data[key]
#                 setattr(args, key, val)
#
#         for key, value in anim_args.__dict__.items():
#             if key in deforum_data:
#                 if deforum_data[key] == "" and "schedule" not in key:
#                     val = None
#                 else:
#                     val = deforum_data[key]
#                 setattr(anim_args, key, val)
#
#         for key, value in video_args.__dict__.items():
#             if key in deforum_data:
#                 if deforum_data[key] == "" and "schedule" not in key:
#                     val = None
#                 else:
#                     val = deforum_data[key]
#                 setattr(anim_args, key, val)
#
#         for key, value in root.__dict__.items():
#             if key in deforum_data:
#                 if deforum_data[key] == "":
#                     val = None
#                 else:
#                     val = deforum_data[key]
#                 setattr(root, key, val)
#
#         for key, value in loop_args.__dict__.items():
#             if key in deforum_data:
#                 if deforum_data[key] == "":
#                     val = None
#                 else:
#                     val = deforum_data[key]
#                 setattr(loop_args, key, val)
#
#         success = None
#         root.timestring = time.strftime('%Y%m%d%H%M%S')
#         args.timestring = root.timestring
#         args.strength = max(0.0, min(1.0, args.strength))
#
#         root.animation_prompts = deforum_data.get("prompts", {})
#
#
#         if not args.use_init and not anim_args.hybrid_use_init_image:
#             args.init_image = None
#
#         elif anim_args.animation_mode == 'Video Input':
#             args.use_init = True
#
#         current_arg_list = [args, anim_args, video_args, parseq_args, root]
#         full_base_folder_path = os.path.join(os.getcwd(), "output/deforum")
#
#         args.batch_name = f"aiNodes_Deforum_{args.timestring}"
#         args.outdir = os.path.join(full_base_folder_path, args.batch_name)
#
#         root.raw_batch_name = args.batch_name
#         args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
#
#         # os.makedirs(args.outdir, exist_ok=True)
#
#         def generate(*args, **kwargs):
#             from .deforum_comfy_sampler import sample_deforum
#             image = sample_deforum(model, clip, vae, **kwargs)
#
#             return image
#
#         self.deforum = DeforumAnimationPipeline(generate)
#
#         self.deforum.config_dir = os.path.join(os.getcwd(), "output/_deforum_configs")
#         os.makedirs(self.deforum.config_dir, exist_ok=True)
#         # self.deforum.generate_inpaint = self.generate_inpaint
#         import comfy
#         pbar = comfy.utils.ProgressBar(deforum_data["max_frames"])
#
#         def datacallback(data=None):
#             if data:
#                 if "image" in data:
#                     pbar.update_absolute(data["frame_idx"], deforum_data["max_frames"], ("JPEG", data["image"], 512))
#
#         self.deforum.datacallback = datacallback
#         deforum_data["turbo_steps"] = deforum_data["diffusion_cadence"]
#         animation = self.deforum(**deforum_data)
#
#         results = []
#         for i in self.deforum.images:
#             tensor = torch.from_numpy(np.array(i).astype(np.float32) / 255.0)
#
#             results.append(tensor)
#             result = torch.stack(results, dim=0)
#
#         return (result,)


def get_current_keys(anim_args, seed, root, parseq_args=None, video_args=None):
    use_parseq = False if parseq_args == None else True
    anim_args.max_frames += 2
    keys = DeforumAnimKeys(anim_args, seed)  # if not use_parseq else ParseqAnimKeys(parseq_args, video_args)

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
    anim_args.max_frames -= 2
    return keys, prompt_series


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
                "latent": ("LATENT",)
            }
        }

    RETURN_TYPES = (("LATENT",))
    FUNCTION = "cache_it"
    CATEGORY = f"deforum_sampling"
    display_name = "Deforum Cache Latent"
    OUTPUT_NODE = True

    def cache_it(self, latent=None):
        global deforum_cache
        deforum_cache["latent"] = latent
        return (latent,)


class DeforumGetCachedLatentNode:

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = (("LATENT",))
    FUNCTION = "get_cached_latent"
    CATEGORY = f"deforum_data"
    OUTPUT_NODE = True
    display_name = "Deforum Load Cached Latent"

    def get_cached_latent(self):
        latent = deforum_cache.get("latent")
        return (latent,)


class DeforumIteratorNode:

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "deforum_data": ("deforum_data",)
            },
            "optional": {
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "reset":("INT", {"default": 0, "min": 0, "max": 1},)
            }
        }

    RETURN_TYPES = (("DEFORUM_FRAME_DATA", "LATENT", "STRING", "STRING"))
    RETURN_NAMES = (("deforum_frame_data", "latent", "positive_prompt", "negative_prompt"))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum_data"
    display_name = "Deforum Iterator Node"
    frame_index = 0
    seed = ""
    seeds = []

    @torch.inference_mode()
    def get(self, deforum_data, latent=None, seed=None, reset=0, *args, **kwargs):

        reset = True if reset == 1 else False

        root_dict = RootArgs()
        args_dict = {key: value["value"] for key, value in DeforumArgs().items()}
        anim_args_dict = {key: value["value"] for key, value in DeforumAnimArgs().items()}
        output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}
        loop_args_dict = {key: value["value"] for key, value in LoopArgs().items()}
        root = SimpleNamespace(**root_dict)
        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)
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

        keys, prompt_series = get_current_keys(anim_args, args.seed, root)

        if self.frame_index > anim_args.max_frames:
            # self.reset_iteration()
            self.frame_index = 0
            # .should_run = False
            # return [None]
        # else:
        args.scale = keys.cfg_scale_schedule_series[self.frame_index]
        if prompt_series:
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
            for i in range(current_index + 1, len(prompt_series)):
                if prompt_series[i] != prompt_series[current_index]:
                    return i
            return len(prompt_series) - 1  # default to the end if no change found

        if prompt_series:
            last_prompt_change = find_last_prompt_change(self.frame_index, prompt_series)
            next_prompt_change = find_next_prompt_change(self.frame_index, prompt_series)

            distance_between_changes = next_prompt_change - last_prompt_change
            current_distance_from_last = self.frame_index - last_prompt_change

            # Generate blend values for the distance between prompt changes
            blend_values = generate_blend_values(distance_between_changes, blend_type="exponential")

            # Fetch the blend value based on the current frame's distance from the last prompt change
            blend_value = blend_values[current_distance_from_last]
            next_prompt = prompt_series[next_prompt_change]

        gen_args = self.get_current_frame(args, anim_args, root, keys, self.frame_index)

        # self.content.frame_slider.setMaximum(anim_args.max_frames - 1)

        self.args = args
        self.root = root
        if prompt_series:
            gen_args["next_prompt"] = next_prompt
            gen_args["prompt_blend"] = blend_value
        gen_args["frame_index"] = self.frame_index
        gen_args["max_frames"] = anim_args.max_frames

        if latent is None or reset:
            self.rng = ImageRNGNoise((4, args.height // 8, args.width // 8), [self.seed], [self.seed - 1],
                                     0.6, 1024, 1024)

            # if latent == None:

            l = self.rng.first().half()
            latent = {"samples": l}
            gen_args["denoise"] = 1.0
        # else:
        #
        #     latent = self.getInputData(1)
        #     #latent = self.rng.next().half()
        print(f"[ Deforum Iterator: {self.frame_index} / {anim_args.max_frames} {self.seed}]")
        self.frame_index += 1
        # self.content.set_frame_signal.emit(self.frame_index)
        # print(latent)
        # print(f"[ Current Seed List: ]\n[ {self.seeds} ]")

        gen_args["seed"] = int(seed)

        return [gen_args, latent, gen_args["prompt"], gen_args["negative_prompt"], {"ui": {"string": str(self.frame_index)}}]

    def get_current_frame(self, args, anim_args, root, keys, frame_idx):
        prompt, negative_prompt = split_weighted_subprompts(args.prompt, frame_idx, anim_args.max_frames)
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
                "args": args}


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
    display_name = "Deforum KSampler"
    CATEGORY = "sampling"

    def sample(self, model, latent, positive, negative, deforum_frame_data):
        from nodes import common_ksampler

        seed = deforum_frame_data.get("seed", 0)
        steps = deforum_frame_data.get("steps", 10)
        cfg = deforum_frame_data.get("cfg", 7.5)
        sampler_name = deforum_frame_data.get("sampler_name", "euler_a")
        scheduler = deforum_frame_data.get("scheduler", "normal")
        # positive = deforum_frame_data.get("positive")
        # negative = deforum_frame_data.get("negative")
        # latent_image = deforum_frame_data.get("latent_image")
        denoise = deforum_frame_data.get("denoise", 1.0)
        print("DENOISE", denoise)
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,
                               denoise=denoise)


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

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fn"
    display_name = "Deforum Frame Warp"
    CATEGORY = "sampling"

    depth_model = None
    algo = ""

    def fn(self, image, deforum_frame_data):
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

            warped_np_img, self.depth, mask = anim_frame_warp(np_image, args, anim_args, keys, frame_idx,
                                                              depth_model=self.depth_model, depth=None, device='cuda',
                                                              half_precision=True)
            num_channels = len(self.depth.shape)

            if num_channels <= 3:
                depth_image = self.depth_model.to_image(self.depth.detach().cpu())
            else:
                depth_image = self.depth_model.to_image(self.depth[0].detach().cpu())

            ret_depth = pil2tensor(depth_image).detach().cpu()
            # if gs.vram_state in ["low", "medium"] and self.depth_model is not None:
            #     self.depth_model.to('cpu')
            image = Image.fromarray(cv2.cvtColor(warped_np_img, cv2.COLOR_BGR2RGB))

            tensor = pil2tensor(image)

            if mask is not None:
                mask = mask.detach().cpu()
                # mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
                mask = mask.mean(dim=0, keepdim=False)
                mask[mask > 1e-05] = 1
                mask[mask < 1e-05] = 0
                mask = mask[0].unsqueeze(0)

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
            return (tensor, ret_depth,)
            # return [data, tensor, mask, ret_depth, self.depth_model]
        else:
            return (image, image,)


class DeforumColorMatchNode:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fn"
    display_name = "Deforum Color Match"
    CATEGORY = "sampling"

    depth_model = None
    algo = ""

    color_match_sample = None

    def fn(self, image, deforum_frame_data):
        if image is not None:
            anim_args = deforum_frame_data.get("anim_args")
            image = np.array(tensor2pil(image))
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
    display_name = "Deforum Add Noise"
    CATEGORY = "sampling"

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
    prev_image = None
    flow = None
    image_size = None
    methods = ['RAFT', 'DIS Medium', 'DIS Fine', 'Farneback']

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
    display_name = "Deforum Hybrid Motion"
    CATEGORY = "sampling"

    def fn(self, image, hybrid_image, deforum_frame_data, hybrid_method):
        if self.raft_model is None:
            self.raft_model = RAFT()
        #
        # data = self.getInputData(0)
        # image = self.getInputData(1)
        # image_2 = self.getInputData(2)

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


class DeforumLoadVideo:
    # @classmethod
    # def INPUT_TYPES(cls):
    #     input_dir = folder_paths.get_input_directory()
    #     files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    #     video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov'))]  # Add more video formats as needed
    #     return {"required":
    #                 {"video": (sorted(video_files), {"file_upload": True})},
    #             }
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "video"
    display_name = "Deforum Load Video"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_video_frame"

    def __init__(self):
        self.cap = None
        self.current_frame = None

    def load_video_frame(self, image):
        video_path = folder_paths.get_annotated_filepath(image)

        # Initialize or reset video capture
        if self.cap is None or self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
            self.cap = cv2.VideoCapture(video_path)
            self.current_frame = -1

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
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid video file: {}".format(image)
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


class DeforumVideoSaveNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
    images = []
    size = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     "filename_prefix": ("STRING",{"default":"deforum_"}),
                     "fps": ("INT", {"default": 24, "min": 1, "max": 10000},),

                     }
                }

    RETURN_TYPES = ()
    OUTPUT_NODE = True

    FUNCTION = "fn"
    display_name = "Deforum Save Video"
    CATEGORY = "sampling"

    def fn(self, image, deforum_frame_data, filename_prefix, fps):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir)

        counter = find_next_index(full_output_folder, filename_prefix)

        frame_idx = deforum_frame_data["frame_idx"]
        max_frames = deforum_frame_data["anim_args"].max_frames

        # Convert tensor to PIL Image and then to numpy array
        pil_image = tensor2pil(image)

        size = pil_image.size
        if size != self.size:
            self.size = size
            self.images.clear()

        self.images.append(np.array(pil_image).astype(np.uint8))
        print(f"[DEFORUM VIDEO SAVE NODE] holding {len(self.images)} images")
        # When the current frame index reaches the last frame, save the video
        if len(self.images) >= max_frames:  # frame_idx is 0-based

            output_path = os.path.join(full_output_folder, f"{filename}_{counter}.mp4")
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=10, pixelformat='yuv420p')

            for frame in tqdm(self.images, desc="Saving MP4 (imageio)"):
                writer.append_data(frame)
            writer.close()

            self.images = []  # Empty the list for next use
        return (image,)
    @classmethod
    def IS_CHANGED(s, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")


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
    prompt = None
    n_prompt = None
    cond = None
    n_cond = None
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
    display_name = "Deforum Blend Conditionings"
    CATEGORY = "sampling"
    def fn(self, clip, deforum_frame_data, blend_method):
        prompt = deforum_frame_data.get("prompt", "")
        negative_prompt = deforum_frame_data.get("negative_prompt", "")
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
        n_cond = self.get_conditioning(prompt=negative_prompt, clip=clip)

        return (cond, n_cond,)

    def get_conditioning(self, prompt="", clip=None, progress_callback=None):

        """if gs.loaded_models["loaded"] == []:
            for node in self.scene.nodes:
                if isinstance(node, TorchLoaderNode):
                    node.evalImplementation()
                    #print("Node found")"""

        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]


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
        NODE_DISPLAY_NAME_MAPPINGS[name] = display_name