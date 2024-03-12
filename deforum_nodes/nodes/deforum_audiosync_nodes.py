import math

import pandas as pd
import scipy.signal
import scipy.ndimage

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

    CATEGORY = "deforum/audio"

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

def xor(a, b):
    return bool(a) ^ bool(b)

class DeforumAmplitudeToKeyframeSeriesNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"type_name": (schedule_types,),
                    "amplitude": ("AMPLITUDE",),
                     },
                "optional":
                    {
                        "max_frames": ("INT", {"default": 500, "min": 1, "max": 16500, "step": 1}),
                        "math": ("STRING", {"default": "x/100"}),
                        "filter_window": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                        "deforum_frame_data": ("DEFORUM_FRAME_DATA",)
                    }
                }

    RETURN_TYPES = ("DEFORUM_FRAME_DATA", "AMPLITUDE", "STRING")
    FUNCTION = "convert"
    display_name = "Amplitude to Schedule"
    CATEGORY = "deforum/audio"

    def safe_eval(self, expr, t, x, max_f):
        allowed_locals = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
            "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
            "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
            "abs": math.fabs, "pow": math.pow, "floor": math.floor, "ceil": math.ceil,
            "round": round, "min": min, "max": max, "pi": math.pi, "e": math.e,
            "factorial": math.factorial,
            "xor": xor,  # Add custom xor function
            "t": t, "x": x, "max_f": max_f,
            # Adding boolean operations as lambda functions to simulate 'and', 'or', 'not'
            "and": lambda a, b: a and b,
            "or": lambda a, b: a or b,
            "not": lambda a: not a,
        }
        try:
            # Directly using 'if else' in expr is supported by Python syntax
            # Logical operations 'and', 'or', 'not', and 'xor' are supported via the above definitions
            return eval(expr, {"__builtins__": {}}, allowed_locals)
        except NameError as e:
            raise ValueError(f"Invalid expression: {e}")
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {repr(e)}")

    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        return float("NaN")

    def convert(self, type_name, amplitude, max_frames=1500, math="x/100", filter_window=0, deforum_frame_data={}):
        max_f = int(max_frames)

        # Optionally apply a smoothing filter to the amplitude data
        if filter_window > 0:
            amplitude_smoothed = np.convolve(amplitude, np.ones(filter_window) / filter_window, mode='same')
        else:
            amplitude_smoothed = amplitude

        frame_index = 0
        modified_amplitude_list = []
        for x in amplitude_smoothed:
            modified_value = self.safe_eval(math, frame_index, x, max_f)
            modified_amplitude_list.append(modified_value)
            frame_index += 1

        # Ensure the modified amplitude list can still be used as an amplitude input
        modified_amplitude_series = pd.Series(modified_amplitude_list)

        # Format the series for visualization or further processing
        formatted_strings = [f"{idx}:({val})" for idx, val in modified_amplitude_series.items()]
        formatted_string = ", ".join(formatted_strings[:-1]) + " and " + formatted_strings[-1] if len(formatted_strings) > 1 else formatted_strings[0]

        # Update deforum_frame_data if necessary
        if "keys" in deforum_frame_data:
            deforum_frame_data["keys"][f"{type_name}_series"] = modified_amplitude_series.to_dict()

        return (deforum_frame_data, modified_amplitude_series.to_numpy(), formatted_string,)

# class DeforumAmplitudeToKeyframeSeriesNode:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required":
#                     {"type_name": (schedule_types,),
#                      "amplitude": ("AMPLITUDE",),
#                      },
#                 "optional":
#                     {
#                         "max_frames": ("INT",{"default":500, "min":1, "max":16500, "step":1},),
#                         "math": ("STRING", {"default":"x/1000"}),
#                         "deforum_frame_data": ("DEFORUM_FRAME_DATA",)
#                     }
#                 }
#
#     RETURN_TYPES = ("DEFORUM_FRAME_DATA", "AMPLITUDE", "STRING")
#     # RETURN_NAMES = ("POSITIVE", "NEGATIVE")
#     FUNCTION = "convert"
#     display_name = "Amplitude to Schedule"
#     CATEGORY = "deforum"
#     def safe_eval(self, expr, t, x, max_f):
#         # Allowed functions and variables
#         # Allowed functions and variables now include 'min' and 'max'
#         allowed_locals = {
#             "sin": math.sin,
#             "cos": math.cos,
#             "tan": math.tan,
#             "sqrt": math.sqrt,
#             "exp": math.exp,
#             "log": math.log,
#             "abs": math.fabs,
#             "min": min,
#             "max": max,
#             "t": t,  # Current frame index
#             "x": x,  # Current amplitude value
#             "max_f": max_f,  # Max frames
#         }
#
#         # Evaluate the expression safely
#         try:
#             return eval(expr, {"__builtins__": {}}, allowed_locals)
#         except NameError as e:
#             raise ValueError(f"Invalid expression: {e}")
#         except Exception as e:
#             raise ValueError(f"Error evaluating expression: {e}")
#
#     @classmethod
#     def IS_CHANGED(self, *args, **kwargs):
#         # Force re-evaluation of the node
#         return float("NaN")
#
#     def convert(self, type_name, amplitude, max_frames=1500, math="x/100", deforum_frame_data={}):
#         max_f = int(max_frames)
#
#         # Apply the math expression to each element of the amplitude series
#         frame_index = 0
#         modified_amplitude_list = []
#         # Apply the math expression to each element of the amplitude list
#         for x in amplitude:
#             modified_value = self.safe_eval(math, frame_index, x, max_f)
#             modified_amplitude_list.append(modified_value)
#             frame_index += 1
#         modified_amplitude_series = pd.Series(modified_amplitude_list)
#         # Convert the series to a string format
#         formatted_strings = [f"{idx}:({val})" for idx, val in modified_amplitude_series.items()]
#
#         # Join all but the last with commas, and add "and" before the last element
#         formatted_string = ", ".join(formatted_strings[:-1]) + " and " + formatted_strings[-1] if len(
#             formatted_strings) > 1 else formatted_strings[0]
#
#         if "keys" in deforum_frame_data:
#             setattr(deforum_frame_data["keys"], f"{type_name}_series", modified_amplitude_series)
#         return (deforum_frame_data, modified_amplitude_list,formatted_string,)


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
    CATEGORY = "deforum/audio"


    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        # Force re-evaluation of the node
        return float("NaN")

    def convert(self, amplitude):

        return (str(amplitude),)

class DerivativeOfAmplitude:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "amplitude": ("AMPLITUDE",),
                     },}

    CATEGORY = "deforum/audio"

    RETURN_TYPES = ("AMPLITUDE",)
    RETURN_NAMES = ("amplitude_derivative",)
    FUNCTION = "derive"
    display_name = "Derive Amplitude"


    def derive(self, amplitude,):
        derivative = np.diff(amplitude, prepend=amplitude[0])  # Use np.diff with prepend to maintain array length
        return (derivative,)

class SpectralCentroid:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio_fft": ("AUDIO_FFT",),
                     },}

    CATEGORY = "deforum/audio"

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("spectral_centroid",)
    FUNCTION = "calculate"
    display_name = "Amplitude Spectral Centoid"


    def calculate(self, audio_fft,):
        magnitudes = np.abs(audio_fft) / len(audio_fft)
        frequencies = np.linspace(0, audio_fft.sample_rate / 2, len(magnitudes))
        centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
        return (centroid,)

class TimeSmoothing:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "amplitude": ("AMPLITUDE",),
                    },
                "optional": {
                    "window_size": ("INT", {"default": 5, "min": 1}),
                    }
                }

    CATEGORY = "deforum/audio"

    RETURN_TYPES = ("AMPLITUDE",)
    RETURN_NAMES = ("smoothed_amplitude",)
    FUNCTION = "smooth"
    display_name = "Amplitude Time Smoothing"


    def smooth(self, amplitude, window_size,):
        smoothed_amplitude = np.convolve(amplitude, np.ones(window_size)/window_size, mode='same')
        return (smoothed_amplitude,)

class BeatDetection:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio": ("AUDIO",),
                 },
        }

    CATEGORY = "deforum/audio"

    RETURN_TYPES = ("AMPLITUDE",)
    RETURN_NAMES = ("beat_times",)
    FUNCTION = "detect"
    display_name = "Audio to Beat Amplitude"


    def detect(self, audio):
        beat_times = self.find_beat_times(audio)
        # Assuming beat_times are indices, we convert these to time values
        beat_times_in_seconds = beat_times / audio.sample_rate
        beat_times_series = pd.Series(beat_times_in_seconds)
        return (beat_times_series,)

    def find_beat_times(self, audio):
        if audio.num_channels > 1:
            audio_data = audio.get_channel_audio_data(0)
        else:
            audio_data = audio.audio_data

        envelope = self.extract_envelope(audio_data, audio.sample_rate)
        smoothed_envelope = scipy.ndimage.gaussian_filter1d(envelope, sigma=5)
        normalized_envelope = (smoothed_envelope - np.min(smoothed_envelope)) / (np.max(smoothed_envelope) - np.min(smoothed_envelope))
        peaks, _ = scipy.signal.find_peaks(normalized_envelope, height=0.3)  # The height threshold may need adjustment

        return peaks

    def extract_envelope(self, audio_data, sample_rate):
        analytic_signal = scipy.signal.hilbert(audio_data)
        envelope = np.abs(analytic_signal)
        return envelope