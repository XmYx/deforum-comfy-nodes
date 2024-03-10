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

import numpy as np
from scipy.signal import savgol_filter


class ExtractDominantNoteAmplitude:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio_fft": ("AUDIO_FFT",),
        },
            "optional": {
                "min_frequency": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "max_frequency": ("FLOAT", {"default": 8000.0, "min": 0.0, "max": 20000.0, "step": 1.0}),
                "magnitude_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "smoothing_window_size": ("INT", {"default": 5, "min": 1, "max": 101, "step": 2}),
                # Odd number, savgol_filter constraint
            },
        }

    CATEGORY = "AudioAnalysis"

    RETURN_TYPES = ("AMPLITUDE",)
    RETURN_NAMES = ("dominant_note_amplitude",)
    FUNCTION = "extract"

    def extract(self, audio_fft, min_frequency, max_frequency, magnitude_threshold, smoothing_window_size):
        dominant_frequencies_amplitude = []

        for fft_data in audio_fft:
            indices = fft_data.get_indices_for_frequency_bands(min_frequency, max_frequency)
            magnitudes = np.abs(fft_data.fft)[indices]

            # Optional: Apply smoothing to magnitudes
            if smoothing_window_size > 1 and len(magnitudes) > smoothing_window_size:
                try:
                    magnitudes = savgol_filter(magnitudes, smoothing_window_size, 3)  # window size, polynomial order
                except ValueError:
                    # In case the smoothing_window_size is inappropriate for the data length
                    pass

            # Apply magnitude threshold
            magnitudes[magnitudes < magnitude_threshold] = 0

            # Identify dominant frequency index and its amplitude
            if magnitudes.size > 0:  # Ensure there's data after filtering
                dominant_index = np.argmax(magnitudes)
                dominant_frequencies_amplitude.append(magnitudes[dominant_index])
            else:
                dominant_frequencies_amplitude.append(0)

        # Convert to numpy array for consistency
        dominant_frequencies_amplitude = np.array(dominant_frequencies_amplitude)

        return (dominant_frequencies_amplitude,)


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
                        "math": ("STRING", {"default":"x/1000"})
                    }
                }

    RETURN_TYPES = ("DEFORUM_FRAME_DATA", "AMPLITUDE", "STRING")
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
        # Convert the series to a string format
        formatted_strings = [f"{idx}:({val})" for idx, val in modified_amplitude_series.items()]

        # Join all but the last with commas, and add "and" before the last element
        formatted_string = ", ".join(formatted_strings[:-1]) + " and " + formatted_strings[-1] if len(
            formatted_strings) > 1 else formatted_strings[0]

        if "keys" in deforum_frame_data:
            setattr(deforum_frame_data["keys"], f"{type_name}_series", modified_amplitude_series)
        return (deforum_frame_data, modified_amplitude_list,formatted_string,)


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