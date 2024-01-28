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
    OUTPUT_NODE = False
    CATEGORY = f"deforum_data"

    def get(self, deforum_data, *args, **kwargs):

        if deforum_data:
            deforum_data.update(**kwargs.items())
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
class DeforumAnimParamsNode(DeforumDataBase):
    params = get_node_params(deforum_anim_params)
    def __init__(self):
        super().__init__()
    @classmethod
    def INPUT_TYPES(s):
        return s.params

NODE_CLASS_MAPPINGS = {
    "DeforumBaseData": DeforumBaseParamsNode,
    "DeforumAnimData": DeforumAnimParamsNode,
    "DeforumTranslationData": DeforumTranslationParamsNode,
    "DeforumDepthData": DeforumDepthParamsNode,
    "DeforumNoiseParamsData": DeforumNoiseParamsNode,
    "DeforumColorParamsData": DeforumColorParamsNode,
    "DeforumDiffusionParamsData": DeforumDiffusionParamsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeforumBaseData": "Deforum Base Data",
    "DeforumAnimData": "Deforum Anim Data",
    "DeforumTranslationData": "Deforum Translation Data",
    "DeforumDepthData": "Deforum Depth Data",
    "DeforumNoiseParamsData": "Deforum Noise Data",
    "DeforumColorParamsData": "Deforum Color Data",
    "DeforumDiffusionParamsData": "Deforum Diffusion Data",
}