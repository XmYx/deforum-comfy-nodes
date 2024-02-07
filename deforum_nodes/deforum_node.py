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
import importlib
import inspect
import json
import math
import os
import secrets
import time
from types import SimpleNamespace

import numexpr
import numpy as np
import pandas as pd
import torch

from deforum import DeforumAnimationPipeline, ImageRNGNoise
from deforum.pipeline_utils import next_seed
from deforum.pipelines.deforum_animation.animation_helpers import DeforumAnimKeys
from deforum.pipelines.deforum_animation.animation_params import RootArgs, DeforumArgs, DeforumAnimArgs, \
    DeforumOutputArgs, LoopArgs, ParseqArgs
from deforum.utils.string_utils import substitute_placeholders, split_weighted_subprompts
from .deforum_ui_data import (deforum_base_params, deforum_anim_params, deforum_translation_params,
                              deforum_cadence_params, deforum_masking_params, deforum_depth_params,
                              deforum_noise_params, deforum_color_coherence_params, deforum_diffusion_schedule_params,
                              deforum_hybrid_video_params, deforum_video_init_params, deforum_image_init_params,
                              deforum_hybrid_video_schedules)
from .deforum_node_base import DeforumDataBase

deforum_cache = {}

def parse_widget(widget_info:dict) -> tuple:

    parsed_widget = None
    #for key, value in widget_info.items():
    t = widget_info["type"]
    if t == "dropdown":
        parsed_widget = (widget_info["choices"],)
    elif t == "checkbox":
        parsed_widget = ("BOOLEAN", { "default": widget_info['default'] })
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
    data_info["optional"] = {"deforum_data":("deforum_data",)}
    return data_info



class DeforumBaseParamsNode(DeforumDataBase):
    params = get_node_params(deforum_base_params)
    def __init__(self):
        super().__init__()
    @classmethod
    def INPUT_TYPES(s):
        return s.params

class DeforumAnimParamsNode(DeforumDataBase):
    params = get_node_params(deforum_anim_params)
    def __init__(self):
        super().__init__()
    @classmethod
    def INPUT_TYPES(s):
        return s.params

class DeforumTranslationParamsNode(DeforumDataBase):
    params = get_node_params(deforum_translation_params)
    def __init__(self):
        super().__init__()
    @classmethod
    def INPUT_TYPES(s):
        return s.params
class DeforumDepthParamsNode(DeforumDataBase):
    params = get_node_params(deforum_depth_params)
    def __init__(self):
        super().__init__()
    @classmethod
    def INPUT_TYPES(s):
        return s.params
class DeforumNoiseParamsNode(DeforumDataBase):
    params = get_node_params(deforum_noise_params)
    def __init__(self):
        super().__init__()
    @classmethod
    def INPUT_TYPES(s):
        return s.params
class DeforumColorParamsNode(DeforumDataBase):
    params = get_node_params(deforum_color_coherence_params)
    def __init__(self):
        super().__init__()
    @classmethod
    def INPUT_TYPES(s):
        return s.params
class DeforumDiffusionParamsNode(DeforumDataBase):
    params = get_node_params(deforum_diffusion_schedule_params)
    def __init__(self):
        super().__init__()
    @classmethod
    def INPUT_TYPES(s):
        return s.params

class DeforumCadenceParamsNode(DeforumDataBase):
    params = get_node_params(deforum_cadence_params)
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
            deforum_data = {"prompts":prompts}
        return (deforum_data,)

class DeforumSampleNode:
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
    CATEGORY = f"deforum_data"

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

        print("ROOT ANIM PROMPTS", root.animation_prompts)

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

        #os.makedirs(args.outdir, exist_ok=True)

        def generate(*args, **kwargs):
            from .deforum_comfy_sampler import sample_deforum
            image = sample_deforum(model, clip, vae, **kwargs)

            return image
        self.deforum = DeforumAnimationPipeline(generate)

        self.deforum.config_dir = os.path.join(os.getcwd(),"output/_deforum_configs")
        os.makedirs(self.deforum.config_dir, exist_ok=True)
        # self.deforum.generate_inpaint = self.generate_inpaint
        import comfy
        pbar = comfy.utils.ProgressBar(deforum_data["max_frames"])

        def datacallback(data=None):
            if data:
                if "image" in data:
                    print("DEFORUM DATACALLBACK")
                    pbar.update_absolute(data["frame_idx"], deforum_data["max_frames"], ("JPEG", data["image"], 512))

        self.deforum.datacallback = datacallback
        deforum_data["turbo_steps"] = deforum_data["diffusion_cadence"]
        animation = self.deforum(**deforum_data)

        results = []
        for i in self.deforum.images:

            tensor = torch.from_numpy(np.array(i).astype(np.float32) / 255.0)

            results.append(tensor)
            result = torch.stack(results, dim=0)

        return (result,)

def get_current_keys(anim_args, seed, root, parseq_args=None, video_args=None):

    use_parseq = False if parseq_args == None else True
    anim_args.max_frames += 2
    keys = DeforumAnimKeys(anim_args, seed) # if not use_parseq else ParseqAnimKeys(parseq_args, video_args)

    # Always enable pseudo-3d with parseq. No need for an extra toggle:
    # Whether it's used or not in practice is defined by the schedules
    if use_parseq:
        anim_args.flip_2d_perspective = True
        # expand prompts out to per-frame
    if use_parseq and keys.manages_prompts():
        prompt_series = keys.prompts
    else:
        prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames + 1)])
        for i, prompt in root.animation_prompts.items():
            if str(i).isdigit():
                prompt_series[int(i)] = prompt
            else:
                prompt_series[int(numexpr.evaluate(i))] = prompt
        prompt_series = prompt_series.ffill().bfill()
    prompt_series = prompt_series
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
    CATEGORY = f"deforum_data"
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
        return {"required":{}}

    RETURN_TYPES = (("LATENT",))
    FUNCTION = "get_cached_latent"
    CATEGORY = f"deforum_data"
    OUTPUT_NODE = True
    display_name = "Deforum Load Cached Latent"


    def get_cached_latent(self):
        latent = deforum_cache.get("latent")
        print("deforum cached latent", latent)

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
            }
        }
    RETURN_TYPES = (("DEFORUM_FRAME_DATA", "LATENT", "STRING", "STRING"))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum_data"
    display_name = "Deforum Iterator Node"
    frame_index = 0
    seed = ""
    seeds = []

    @torch.inference_mode()
    def get(self, deforum_data, latent=None, seed=None, *args, **kwargs):

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
        #print(anim_args.max_frames)

        root.animation_prompts = deforum_data.get("prompts")

        keys, prompt_series = get_current_keys(anim_args, args.seed, root)
        # print(f"WOULD RETURN\n{keys}\n\n{prompt_series}")

        if self.frame_index > anim_args.max_frames:
            #self.reset_iteration()
            self.frame_index = 0
            #.should_run = False
            #return [None]
        # else:
        args.scale = keys.cfg_scale_schedule_series[self.frame_index]
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

        # Inside your main loop:

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

        #self.content.frame_slider.setMaximum(anim_args.max_frames - 1)

        self.args = args
        self.root = root
        gen_args["next_prompt"] = next_prompt
        gen_args["prompt_blend"] = blend_value
        gen_args["frame_index"] = self.frame_index
        gen_args["max_frames"] = anim_args.max_frames


        if self.frame_index == 0:
            self.rng = ImageRNGNoise((4, args.height // 8, args.width // 8), [self.seed], [self.seed - 1],
                                0.6, 1024, 1024)

            # if latent == None:

            l = self.rng.first().half()
            latent = {"samples":l}
        # else:
        #
        #     latent = self.getInputData(1)
        #     #latent = self.rng.next().half()
        print(f"[ Deforum Iterator: {self.frame_index} / {anim_args.max_frames} {self.seed}]")
        self.frame_index += 1
        #self.content.set_frame_signal.emit(self.frame_index)
        #print(latent)
        #print(f"[ Current Seed List: ]\n[ {self.seeds} ]")

        gen_args["seed"] = int(seed)

        return [gen_args, latent, gen_args["prompt"], gen_args["negative_prompt"]]


    def get_current_frame(self, args, anim_args, root, keys, frame_idx):
        prompt, negative_prompt = split_weighted_subprompts(args.prompt, frame_idx, anim_args.max_frames)
        strength = keys.strength_schedule_series[frame_idx] if not frame_idx == 0 or args.use_init else 1.0
        return {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "denoise": strength,
                "cfg": args.scale,
                "steps": int(keys.steps_schedule_series[self.frame_index]),
                "root": root,
                "keys": keys,
                "frame_idx": frame_idx}


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

        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise)

# NODE_CLASS_MAPPINGS = {
#     "DeforumBaseData": DeforumBaseParamsNode,
#     "DeforumAnimData": DeforumAnimParamsNode,
#     "DeforumTranslationData": DeforumTranslationParamsNode,
#     "DeforumDepthData": DeforumDepthParamsNode,
#     "DeforumNoiseParamsData": DeforumNoiseParamsNode,
#     "DeforumColorParamsData": DeforumColorParamsNode,
#     "DeforumDiffusionParamsData": DeforumDiffusionParamsNode,
#     "DeforumCadenceParams": DeforumCadenceParamsNode,
#     "DeforumPrompt": DeforumPromptNode,
#     "DeforumSampler": DeforumSampleNode,
#     "DeforumIterator": DeforumIteratorNode,
#     "DeforumCacheWrite": DeforumCacheLatentNode,
#     "DeforumCacheLoad": DeforumGetCachedLatentNode,
# }
#
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "DeforumBaseData": "Deforum Base Data",
#     "DeforumAnimData": "Deforum Anim Data",
#     "DeforumTranslationData": "Deforum Translation Data",
#     "DeforumDepthData": "Deforum Depth Data",
#     "DeforumNoiseParamsData": "Deforum Noise Data",
#     "DeforumColorParamsData": "Deforum Color Data",
#     "DeforumCadenceParams": "Deforum Cadence Data",
#     "DeforumPrompt": "Deforum Prompts",
#     "DeforumSampler": "Deforum Sampler",
#     "DeforumIterator": "Deforum Iterator",
#     "DeforumCacheWrite": "Deforum Write Cache Latent",
#     "DeforumCacheLoad": "Deforum Load Cache Latent",
# }

# Create an empty dictionary for class mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Iterate through all classes defined in your code
# Import the deforum_nodes.deforum_node module
deforum_node_module = importlib.import_module('deforum-comfy-nodes.deforum_nodes.deforum_node')

# Iterate through all classes defined in deforum_nodes.deforum_node
for name, obj in inspect.getmembers(deforum_node_module):

    print(name, obj)

    # Check if the member is a class
    if inspect.isclass(obj) and hasattr(obj, "INPUT_TYPES"):
        # Extract the class name and display name
        class_name = name
        display_name = getattr(obj, "display_name", name)  # Use class attribute or default to class name
        # Add the class to the mappings
        NODE_CLASS_MAPPINGS[class_name] = obj
        NODE_DISPLAY_NAME_MAPPINGS[name] = display_name