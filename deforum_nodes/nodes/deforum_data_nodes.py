from ..modules.deforum_ui_data import (deforum_base_params, deforum_anim_params, deforum_translation_params,
                                       deforum_cadence_params, deforum_depth_params,
                                       deforum_noise_params, deforum_color_coherence_params, deforum_diffusion_schedule_params,
                                       deforum_hybrid_video_params, deforum_hybrid_video_schedules)
from ..modules.deforum_node_base import DeforumDataBase
from ..modules.deforum_comfyui_helpers import get_node_params

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
        seed = deforum_frame_data.get("seed", 0)
        steps = deforum_frame_data.get("steps", 10)
        cfg = deforum_frame_data.get("cfg", 7.5)
        sampler_name = deforum_frame_data.get("sampler_name", "euler_a")
        scheduler = deforum_frame_data.get("scheduler", "normal")
        denoise = deforum_frame_data.get("denoise", 1.0)

        keys = deforum_frame_data.get("keys")
        frame_idx = deforum_frame_data.get("frame_idx")
        subseed_str = keys.subseed_strength_schedule_series[frame_idx]

        return (frame_idx, seed, steps, cfg, sampler_name, scheduler, denoise,subseed_str,)