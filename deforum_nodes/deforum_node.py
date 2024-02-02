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
import json
import os
import secrets
import time
from types import SimpleNamespace

import numpy as np
import torch

from deforum import DeforumAnimationPipeline
from deforum.pipelines.deforum_animation.animation_helpers import DeforumAnimKeys
from deforum.pipelines.deforum_animation.animation_params import RootArgs, DeforumArgs, DeforumAnimArgs, \
    DeforumOutputArgs, LoopArgs, ParseqArgs
from deforum.utils.string_utils import substitute_placeholders, split_weighted_subprompts
from .deforum_ui_data import (deforum_base_params, deforum_anim_params, deforum_translation_params,
                              deforum_cadence_params, deforum_masking_params, deforum_depth_params,
                              deforum_noise_params, deforum_color_coherence_params, deforum_diffusion_schedule_params,
                              deforum_hybrid_video_params, deforum_video_init_params, deforum_image_init_params,
                              deforum_hybrid_video_schedules)

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
    data_info["optional"] = {"deforum_data":("DEFORUM_DATA",)}
    return data_info

class DeforumDataBase:

    @classmethod
    def INPUT_TYPES(s):
        return s.params

    RETURN_TYPES = (("deforum_data",))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum_data"

    def get(self, deforum_data=None, *args, **kwargs):

        if deforum_data:
            deforum_data.update(**kwargs)
        else:
            deforum_data = kwargs
        print(deforum_data)
        return (deforum_data,)

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
            # "optional": {
            #     "text_c": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
            #     "frame_c": ("INT", {"default": 24}),
            #     "text_d": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
            #     "frame_d": ("INT", {"default": 36}),
            #     "text_e": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
            #     "frame_e": ("INT", {"default": 48}),
            #     "text_f": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
            #     "frame_f": ("INT", {"default": 60}),
            #     "text_g": ("STRING", {"forceInput": True, "multiline": True, "default": ""}),
            #     "frame_g": ("INT", {"default": 72})
            # }
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
        #args.prompts = json.loads(args_dict_main['animation_prompts'])
        #args.positive_prompts = args_dict_main['animation_prompts_positive']
        #args.negative_prompts = args_dict_main['animation_prompts_negative']

        if not args.use_init and not anim_args.hybrid_use_init_image:
            args.init_image = None

        elif anim_args.animation_mode == 'Video Input':
            args.use_init = True

        current_arg_list = [args, anim_args, video_args, parseq_args, root]
        full_base_folder_path = os.path.join(os.getcwd(), "output/deforum")
        #root.raw_batch_name = args.batch_name


        args.batch_name = f"aiNodes_Deforum_{args.timestring}"
        args.outdir = os.path.join(full_base_folder_path, args.batch_name)

        root.raw_batch_name = args.batch_name
        args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)

        os.makedirs(args.outdir, exist_ok=True)



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

        # if self.deforum.args.seed == -1 or self.deforum.args.seed == "-1":
        #     setattr(self.deforum.args, "seed", secrets.randbelow(999999999999999999))
        #     setattr(self.deforum.root, "raw_seed", int(self.deforum.args.seed))
        #     setattr(self.deforum.root, "seed_internal", 0)
        # else:
        #     self.deforum.args.seed = int(self.deforum.args.seed)

        #self.deforum.keys = DeforumAnimKeys(self.deforum.gen, self.deforum.gen.seed)

        #deforum_data["prompts"] = {0:"Cat sushi"}
        deforum_data["turbo_steps"] = deforum_data.get("diffusion_cadence", 1)
        deforum_data["store_frames_in_ram"] = True
        animation = self.deforum(**deforum_data)

        results = []
        for i in self.deforum.images:

            tensor = torch.from_numpy(np.array(i).astype(np.float32) / 255.0)

            results.append(tensor)
            result = torch.stack(results, dim=0)

        return (result,)

NODE_CLASS_MAPPINGS = {
    "DeforumBaseData": DeforumBaseParamsNode,
    "DeforumAnimData": DeforumAnimParamsNode,
    "DeforumTranslationData": DeforumTranslationParamsNode,
    "DeforumDepthData": DeforumDepthParamsNode,
    "DeforumNoiseParamsData": DeforumNoiseParamsNode,
    "DeforumColorParamsData": DeforumColorParamsNode,
    "DeforumDiffusionParamsData": DeforumDiffusionParamsNode,
    "DeforumCadenceParams": DeforumCadenceParamsNode,
    "DeforumPrompt": DeforumPromptNode,
    "DeforumSampler": DeforumSampleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeforumBaseData": "Deforum Base Data",
    "DeforumAnimData": "Deforum Anim Data",
    "DeforumTranslationData": "Deforum Translation Data",
    "DeforumDepthData": "Deforum Depth Data",
    "DeforumNoiseParamsData": "Deforum Noise Data",
    "DeforumColorParamsData": "Deforum Color Data",
    "DeforumCadenceParams": "Deforum Cadence Data",
    "DeforumPrompt": "Deforum Prompts",
    "DeforumSampler": "Deforum Sampler",
}
