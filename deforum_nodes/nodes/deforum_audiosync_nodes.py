import math

import pandas as pd

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