import math

import pandas as pd
import scipy.signal
import scipy.ndimage

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import librosa

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

# class DeforumAmplitudeToKeyframeSeriesNode:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required":
#                     {"type_name": (schedule_types,),
#                     "amplitude": ("AMPLITUDE",),
#                      },
#                 "optional":
#                     {
#                         "max_frames": ("INT", {"default": 500, "min": 1, "max": 16500, "step": 1}),
#                         "math": ("STRING", {"default": "x/100"}),
#                         "filter_window": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
#                         "deforum_frame_data": ("DEFORUM_FRAME_DATA",)
#                     }
#                 }
#
#     RETURN_TYPES = ("DEFORUM_FRAME_DATA", "AMPLITUDE", "STRING")
#     FUNCTION = "convert"
#     display_name = "Amplitude to Schedule"
#     CATEGORY = "deforum/audio"
#
#     def safe_eval(self, expr, t, x, max_f):
#         allowed_locals = {
#             "sin": math.sin, "cos": math.cos, "tan": math.tan,
#             "asin": math.asin, "acos": math.acos, "atan": math.atan,
#             "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
#             "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
#             "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
#             "abs": math.fabs, "pow": math.pow, "floor": math.floor, "ceil": math.ceil,
#             "round": round, "min": min, "max": max, "pi": math.pi, "e": math.e,
#             "factorial": math.factorial,
#             "xor": xor,  # Add custom xor function
#             "t": t, "x": x, "max_f": max_f,
#             # Adding boolean operations as lambda functions to simulate 'and', 'or', 'not'
#             "and": lambda a, b: a and b,
#             "or": lambda a, b: a or b,
#             "not": lambda a: not a,
#         }
#         try:
#             # Directly using 'if else' in expr is supported by Python syntax
#             # Logical operations 'and', 'or', 'not', and 'xor' are supported via the above definitions
#             return eval(expr, {"__builtins__": {}}, allowed_locals)
#         except NameError as e:
#             raise ValueError(f"Invalid expression: {e}")
#         except Exception as e:
#             raise ValueError(f"Error evaluating expression: {repr(e)}")
#
#     @classmethod
#     def IS_CHANGED(self, *args, **kwargs):
#         return float("NaN")
#
#     def convert(self, type_name, amplitude, max_frames=1500, math="x/100", filter_window=0, deforum_frame_data={}):
#         max_f = int(max_frames)
#
#         # Optionally apply a smoothing filter to the amplitude data
#         if filter_window > 0:
#             amplitude_smoothed = np.convolve(amplitude, np.ones(filter_window) / filter_window, mode='same')
#         else:
#             amplitude_smoothed = amplitude
#
#         frame_index = 0
#         modified_amplitude_list = []
#         for x in amplitude_smoothed:
#             modified_value = self.safe_eval(math, frame_index, x, max_f)
#             modified_amplitude_list.append(modified_value)
#             frame_index += 1
#
#         # Ensure the modified amplitude list can still be used as an amplitude input
#         modified_amplitude_series = pd.Series(modified_amplitude_list)
#
#         # Format the series for visualization or further processing
#         formatted_strings = [f"{idx}:({val})" for idx, val in modified_amplitude_series.items()]
#         formatted_string = ", ".join(formatted_strings[:-1]) + " and " + formatted_strings[-1] if len(formatted_strings) > 1 else formatted_strings[0]
#
#         # Update deforum_frame_data if necessary
#         if "keys" in deforum_frame_data:
#             deforum_frame_data["keys"][f"{type_name}_series"] = modified_amplitude_series.to_dict()
#
#         return (deforum_frame_data, modified_amplitude_series.to_numpy(), formatted_string,)
import numpy as np
import pandas as pd
import math

class DeforumAmplitudeToKeyframeSeriesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"type_name": (schedule_types,),
                    "amplitude": ("AMPLITUDE",),
                     },
                "optional":
                    {
                        "max_frames": ("INT", {"default": 500, "min": 1, "max": 16500, "step": 1}),
                        "math": ("STRING", {"default": "x/100"}),
                        "filter_window": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                        "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                        "y": ("AMPLITUDE",),
                        "z": ("AMPLITUDE",),
                    }
                }

    RETURN_TYPES = ("DEFORUM_FRAME_DATA", "AMPLITUDE", "STRING")
    FUNCTION = "convert"
    display_name = "Amplitude to Schedule"
    CATEGORY = "deforum/audio"

    def safe_eval(self, expr, t, x, max_f, y=None, z=None):
        allowed_locals = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
            "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
            "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
            "abs": math.fabs, "pow": math.pow, "floor": math.floor, "ceil": math.ceil,
            "round": round, "min": min, "max": max, "pi": math.pi, "e": math.e,
            "factorial": math.factorial,
            "xor": lambda a, b: a ^ b,  # Python's `^` operator for XOR
            "t": t, "x": x, "max_f": max_f, "y": y, "z": z,
            # Boolean operations via lambda functions
            "and": lambda a, b: a and b,
            "or": lambda a, b: a or b,
            "not": lambda a: not a,
        }
        try:
            return eval(expr, {"__builtins__": {}}, allowed_locals)
        except NameError as e:
            raise ValueError(f"Invalid expression: {e}")
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {repr(e)}")

    def convert(self, type_name, amplitude, max_frames=500, math="x/100", filter_window=0, deforum_frame_data={}, y=None, z=None):
        max_f = int(max_frames)

        # Apply smoothing filter to the amplitude data if needed
        amplitude_smoothed = np.convolve(amplitude, np.ones(filter_window) / filter_window, mode='same') if filter_window > 0 else amplitude
        y_smoothed = np.convolve(y, np.ones(filter_window) / filter_window, mode='same') if y is not None and filter_window > 0 else y
        z_smoothed = np.convolve(z, np.ones(filter_window) / filter_window, mode='same') if z is not None and filter_window > 0 else z

        frame_index = 0
        modified_amplitude_list = []
        for idx, x in enumerate(amplitude_smoothed):
            y_val = y_smoothed[idx] if y is not None else None
            z_val = z_smoothed[idx] if z is not None else None
            modified_value = self.safe_eval(math, frame_index, x, max_f, y=y_val, z=z_val)
            modified_amplitude_list.append(modified_value)
            frame_index += 1

        modified_amplitude_series = pd.Series(modified_amplitude_list)
        formatted_strings = [f"{idx}:({val})" for idx, val in modified_amplitude_series.items()]
        formatted_string = ", ".join(formatted_strings[:-1]) + " and " + formatted_strings[-1] if len(formatted_strings) > 1 else formatted_strings[0]

        if "keys" in deforum_frame_data:
            deforum_frame_data["keys"][f"{type_name}_series"] = modified_amplitude_series.to_dict()

        return (deforum_frame_data, modified_amplitude_series.to_numpy(), formatted_string,)


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



class FrequencyRangeAmplitude:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "frequency_range": ("TUPLE", {"default": (20, 20000)}),
                "window_size": ("INT", {"default": 1}),
                "inverted": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "deforum/audio"

    RETURN_TYPES = ("AMPLITUDE",)
    RETURN_NAMES = ("frequency_range_amplitude",)
    FUNCTION = "analyze_frequency_range"
    display_name = "Frequency Range Amplitude"

    def analyze_frequency_range(self, audio, frequency_range=(20, 20000), window_size=1, inverted=False):
        if audio.num_channels > 1:
            audio_data = audio.get_channel_audio_data(0)
        else:
            audio_data = audio.audio_data
        sample_rate = audio.sample_rate
        # Apply bandpass filter
        filtered_audio = self.bandpass_filter(audio_data, frequency_range[0], frequency_range[1], sample_rate)
        # Calculate FFT
        fft_result = np.fft.rfft(filtered_audio)
        fft_freqs = np.fft.rfftfreq(len(filtered_audio), 1 / sample_rate)
        # Extract amplitude
        amplitudes = np.abs(fft_result)
        # Normalize and smooth if necessary
        normalized_amplitudes = self.normalize(amplitudes, window_size=window_size, inverted=inverted)
        return (normalized_amplitudes,)

    def bandpass_filter(self, data, lowcut, highcut, sample_rate, order=5):
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y

    def normalize(self, data, min_value=0, max_value=1, window_size=1, inverted=False):
        # Normalization and smoothing logic as described in the notebook
        # Return normalized data
        pass  # Implement normalization and optional smoothing as described

class BeatDetectionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample_rate": ("INT",{"default": 44100, "min": 8000, "max": 200000, "step":1}),
            }
        }

    CATEGORY = "deforum/audio"

    RETURN_TYPES = ("AMPLITUDE",)
    RETURN_NAMES = ("beat_times",)
    FUNCTION = "detect_beats"
    display_name = "Beat Detection v2"

    def detect_beats(self, audio, sample_rate):
        if audio.num_channels > 1:
            audio_data = audio.get_channel_audio_data(0)
        else:
            audio_data = audio.audio_data
        # Assuming audio_data is already pre-processed to be in the correct format
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        beat_times = librosa.frames_to_time(beats, sr=sample_rate)
        return (beat_times,)

class TempoChangeDetectionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"audio": ("AUDIO",)},
            "optional": {"threshold": ("FLOAT", {"default": 0.5, "min":0.0, "max":1250.0, "step":0.1})},
        }

    CATEGORY = "deforum/audio"
    RETURN_TYPES = ("AMPLITUDE",)
    RETURN_NAMES = ("tempo_change_times",)
    FUNCTION = "detect_tempo_changes"
    display_name = "Tempo Change Detection"

    def detect_tempo_changes(self, audio, threshold=0.5, fps=24.0):
        audio_data = audio.get_channel_audio_data(0) if audio.num_channels > 1 else audio.audio_data
        sample_rate = audio.sample_rate

        # Calculate onset strength
        onset_env = librosa.onset.onset_strength(y=audio_data.astype(np.float32), sr=sample_rate)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sample_rate)
        dynamic_tempo = np.mean(tempogram, axis=1)
        times = librosa.frames_to_time(np.arange(len(dynamic_tempo)), sr=sample_rate)

        # Detect change points
        change_points = np.abs(np.diff(dynamic_tempo)) > threshold
        change_times = times[:-1][change_points]

        # Convert change times to a sequence of amplitudes with the desired fps
        # First, create a boolean array indicating change points
        full_length = int(np.ceil(times[-1] * fps))
        change_points_sequence = np.zeros(full_length, dtype=bool)
        interpolated_times = np.arange(0, times[-1], step=1.0 / fps)

        for time in interpolated_times:
            index = int(time * fps)
            if index < full_length:
                change_points_sequence[index] = True

        # Optionally, fill in missing values if needed, but here we map detected changes directly
        # Return the sequence as desired
        print(len(change_points_sequence))  # Debug print to check the length of the output sequence
        return (change_points_sequence,)