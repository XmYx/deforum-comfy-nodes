import random

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io

from deforum.pipelines.deforum_animation.animation_helpers import FrameInterpolator
from ..modules.deforum_comfyui_helpers import tensor2pil, tensor2np, pil2tensor


templates = [
    "0:(0), max_f:(100)", # Linearly interpolates from 0 to 100.
    "0:(100), max_f:(0)", # Linearly interpolates from 100 to 0.
    "0:(sin(2*3.14*t/max_f))", # Sinusoidal wave across the frame range.
    "0:(exp(-t/max_f)*sin(4*3.14*t/max_f))", # Damped sinusoidal wave.
    "0:(s%100)", # Outputs a constant value based on the seed modulo 100.
    "0:(sin(2*3.14*(t+s)/max_f))", # Sinusoidal wave with phase shift based on seed.
    "0:(t**2/max_f**2)", # Quadratic growth over frames.
    "0:(-t**2/max_f**2 + 1)", # Inverted quadratic growth.
    "0:(0.5*sin(2*3.14*t/max_f) + 0.5*sin(4*3.14*t/max_f))", # Superposition of two sinusoidal waves.
    "0:(0.5*cos(2*3.14*t/max_f) + t/max_f)", # Combination of cosine wave and linear growth.
    "0:(sin(2*3.14*t/max_f + s))", # Sinusoidal wave with phase offset by seed.
    "0:(sin(2*3.14*t/max_f) * (s % 10))", # Sinusoidal wave amplitude modulated by seed.
    "0:(cos(2*3.14*t/max_f))", # Cosine wave over the frame range.
    "0:(1 - abs(2*t/max_f - 1))", # Triangle wave that peaks at mid-frame.
    "0:(sin(2*3.14*t/max_f) * cos(2*3.14*t/max_f))", # Product of sine and cosine waves.
    "0:(t % 20)", # Modulo operation creates a sawtooth wave.
    "0:(50 + 50*sin(2*3.14*t/max_f))", # Sinusoidal oscillation around 50.
    "0:(s % 10 + sin(2*3.14*t/max_f))", # Sinusoidal wave with base level determined by seed.
    "0:(log(t+1))", # Logarithmic growth.
    "0:(-cos(2*3.14*t/max_f) + 2)", # Inverted cosine wave shifted upwards.
    "0:(sin(2*3.14*t/max_f) if t < max_f/2 else cos(2*3.14*t/max_f))", # Sinusoidal that switches to cosine at halfway.
    "0:(sin(4*3.14*t/max_f) + cos(2*3.14*t/max_f))", # Sum of two waves with different frequencies.
    "0:(tan(3.14*t/max_f))", # Tangent wave, potentially extreme values.
    "0:(1 - t/max_f)", # Linear decrease from 1 to 0.
    "0:(sqrt(t/max_f))", # Square root growth.
    "0:(-sqrt(t/max_f) + 1)", # Inverted square root curve.
    "0:(abs(sin(2*3.14*t/max_f)))", # Absolute value of a sinusoidal wave, creating peaks.
    "0:(abs(cos(2*3.14*t/max_f)))", # Absolute value of a cosine wave, creating peaks.
    "0:(sin(2*3.14*t/max_f)**2)", # Square of a sinusoidal wave, smoothing negative values.
    "0:(cos(2*3.14*t/max_f)**2)", # Square of a cosine wave, smoothing negative values.
    "0:(sin(2*3.14*t/(max_f+s)))", # Sinusoidal wave with frequency modulated by seed.
    "0:(exp(t/max_f) - 1)", # Exponential growth.
    "0:(2**(t/max_f) - 1)", # Exponential growth with base 2.
    "0:(-2**(t/max_f) + 2)", # Inverted exponential curve with base 2.
    "0:(s % 10 * t/max_f)", # Linear growth modulated by seed.
    "0:(sin(2*3.14*(t+s)%max_f/max_f))", # Sinusoidal wave with frequency and phase modulated by seed.
    "0:(cos(2*3.14*(t+s)%max_f/max_f))", # Cosine wave with frequency and phase modulated by seed.
    "0:(s * sin(2*3.14*t/max_f))", # Sinusoidal wave with amplitude modulated by seed.
    "0:(cos(t/max_f * 3.14 * (s%5)))", # Cosine wave with frequency modulated by seed.
    "0:(sin(t**2/max_f**2))", # Sinusoidal wave applied to quadratic growth.
    "0:(cos(s + t/max_f))", # Cosine wave with phase shift linearly increasing and offset by seed.
    "0:(tan(s * t/max_f))", # Tangent wave with slope modulated by seed.
    "0:(-1*sin(3.14*t/max_f))", # Inverted sinusoidal wave.
    "0:(exp(-t/max_f)*cos(2*3.14*t/max_f))", # Damped cosine wave.
    "0:(sin(2*3.14*t/max_f)**3)", # Cubed sinusoidal wave for enhanced contrast.
    "0:(cos(2*3.14*t/max_f)**3)", # Cubed cosine wave for enhanced contrast.
    "0:(log(t+1)*sin(2*3.14*t/max_f))", # Sinusoidal wave amplitude modulated by a logarithmic function.
    "0:(sin(t/max_f)*cos(t/max_f))", # Product of sine and cosine for varying wave patterns.
    "0:(t**3/max_f**3)", # Cubic growth for accelerated change.
    "0:(-t**3/max_f**3 + 1)", # Inverted cubic curve for decelerated change towards the end.
    "0:(abs(sin(4*3.14*t/max_f)))", # Absolute sinusoidal wave with higher frequency.
    "0:(abs(cos(4*3.14*t/max_f)))", # Absolute cosine wave with higher frequency.
    "0:(sin(2*3.14*t/max_f)*s)", # Sinusoidal wave with amplitude directly proportional to seed.
    "0:(cos(2*3.14*t/max_f)*s)", # Cosine wave with amplitude directly proportional to seed.
    "0:(tan(2*3.14*t/max_f + s))", # Tangent wave with phase shift based on seed.
    "0:(exp(t/max_f)*sin(2*3.14*t/max_f))", # Exponentially growing sinusoidal wave.
    "0:(2**(t/max_f)*sin(2*3.14*t/max_f))", # Sinusoidal wave with exponentially growing amplitude.
    "0:(-2**(t/max_f)*cos(2*3.14*t/max_f) + 2)", # Exponentially damped cosine wave, shifted up.
    "0:(s%20 * sin(2*3.14*t/max_f))", # Sinusoidal wave with amplitude modulated by seed modulo 20.
    "0:(cos(2*3.14*(t+s)%max_f/max_f)*s)", # Cosine wave with frequency, phase modulated by seed, and amplitude scaling.
    "0:(sin(2*3.14*(t+s)%max_f/max_f)*s)", # Sinusoidal wave with frequency, phase modulated by seed, and amplitude scaling.
    "0:(sin(2*3.14*t/max_f)/(t+1))", # Sinusoidal wave with amplitude inversely proportional to frame.
    "0:(cos(2*3.14*t/max_f)/(t+1))", # Cosine wave with amplitude inversely proportional to frame.
    "0:(sin(2*3.14*t/max_f)*t/max_f)", # Sinusoidal wave with linearly increasing amplitude.
    "0:(cos(2*3.14*t/max_f)*t/max_f)", # Cosine wave with linearly increasing amplitude.
    "0:(sin(2*3.14*t/max_f)*sqrt(t/max_f))", # Sinusoidal wave with amplitude modulated by square root of frame.
    "0:(cos(2*3.14*t/max_f)*sqrt(t/max_f))", # Cosine wave with amplitude modulated by square root of frame.
    "0:(tan(2*3.14*t/max_f)*sqrt(t/max_f))", # Tangent wave with amplitude modulated by square root of frame.
    "0:(sin(2*3.14*t/max_f)*log(t+1))", # Sinusoidal wave with logarithmically increasing amplitude.
    "0:(cos(2*3.14*t/max_f)*log(t+1))", # Cosine wave with logarithmically increasing amplitude.
    "0:(tan(2*3.14*t/max_f)*log(t+1))" # Tangent wave with logarithmically increasing amplitude.
    "0:((exp(t/max_f)) * (log(1+abs(t))) + (exp(1000/max_f)) + 1)",  # Something Random
    "0:((exp(t/max_f)) / (cos(2*3.14*t/max_f)) / (log(1+abs(t))) / (tan(2*3.14*t/max_f)) + 1)",
    "0:((log(1+abs(t))) + (log(1+abs(1000))) - (cos(2*3.14*1000/max_f)) + (cos(2*3.14*t/max_f)) + (sin(2*3.14*1000/max_f)) + 1)"
]

audio_templates = [
    "x * t / max_f",  # Linear scaling of x with respect to t.
    "x * sin(2 * 3.14 * t / max_f)",  # Modulating x with a sinusoidal wave based on frame.
    "x * (1 - t / max_f)",  # Linear decrease of x towards the end.
    "x * sin(2 * 3.14 * t / max_f) + x * cos(2 * 3.14 * t / max_f)",  # Superposition of sine and cosine waves scaled by x.
    "x * log(t + 1) / log(max_f)",  # Logarithmic scaling of x.
    "x * sqrt(t / max_f)",  # Square root scaling of x.
    "x * (exp(t / max_f) - 1)",  # Exponential scaling of x.
    "x * sin(2 * 3.14 * t / max_f) * cos(2 * 3.14 * t / max_f)",  # Product of sine and cosine waves scaled by x.
    "x * (1 - abs(2 * t / max_f - 1))",  # Triangle wave modulation of x.
    "x * abs(sin(2 * 3.14 * t / max_f))",  # Absolute sinusoidal wave scaled by x.
    "x * (2 ** (t / max_f) - 1)",  # Exponential growth base 2 scaled by x.
    "x * (-2 ** (t / max_f) + 2)",  # Inverted exponential curve base 2 scaled by x.
    "x * (sin(2 * 3.14 * t / max_f) if t < max_f / 2 else cos(2 * 3.14 * t / max_f))",  # Conditional wave modulation of x.
    "x * tan(3.14 * t / max_f)",  # Tangent wave scaling of x, watch for extreme values.
    "x * (1 - t / max_f)",  # Linear decrease from x to 0.
    "x * sin(2 * 3.14 * t / max_f)**2",  # Square of sinusoidal wave scaled by x, smoothing negatives.
    "x * cos(2 * 3.14 * t / max_f)**2",  # Square of cosine wave scaled by x, smoothing negatives.
    "x * exp(-t / max_f)",  # Exponential decay of x.
    "x * (s % 10 + sin(2 * 3.14 * t / max_f))",  # Sinusoidal wave with base level determined by seed, scaled by x.
    "x * sin(4 * 3.14 * t / max_f) + x * cos(2 * 3.14 * t / max_f)",  # Sum of two waves with different frequencies scaled by x.
    "x * (sin(2 * 3.14 * t / max_f) * (t % 10))",  # Sinusoidal wave amplitude modulated by frame modulo, scaled by x.
    "x * abs(cos(2 * 3.14 * t / max_f))",  # Absolute value of a cosine wave scaled by x, creating peaks.
    "x * (sin(2 * 3.14 * t / (max_f + t)))",  # Sinusoidal wave with frequency modulated by frame, scaled by x.
    "x * (sin(2 * 3.14 * t / max_f) / (t + 1))",  # Sinusoidal wave with amplitude inversely proportional to frame, scaled by x.
    "(exp(t / max_f)) * x + (log(1 + abs(t))) * x",  # Combination of exponential growth and logarithmic scaling of x.
    "(x * cos(2 * 3.14 * t / max_f)) / (1 + log(t + 1))",  # Cosine wave scaled by x with logarithmic denominator to soften growth.
]

def generate_complex_random_expression(max_frames, seed=None, max_parts=3):
    """
    Generates a complex random mathematical expression using a variety of functions, operators, and globals.

    Parameters:
    - max_frames: The maximum number of frames, for normalizing 't' in expressions.
    - seed: Optional seed for random number generator for reproducibility.
    - max_parts: Maximum number of parts (expressions) to combine.

    Returns:
    A string formatted like "0:(expression), max_f:(expression)".
    """
    if seed is not None:
        random.seed(seed)

    funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'abs']
    operators = ['+', '-', '*', '/']
    globals = ['t', 'max_f', 's']
    parts = []

    for _ in range(random.randint(1, max_parts)):
        func = random.choice(funcs)
        operator = random.choice(operators) if parts else ''  # No leading operator for the first part
        global_var = random.choice(globals)
        if global_var == 'max_f':
            global_value = str(max_frames)
        else:
            global_value = global_var

        if func in ['sin', 'cos', 'tan']:
            part = f"{func}(2*3.14*{global_value}/max_f)"
        elif func == 'log':
            part = f"{func}(1+abs({global_value}))"
        elif func == 'exp':
            part = f"{func}({global_value}/max_f)"
        else:  # abs or other functions without specific handling
            part = f"{func}({global_value})"

        parts.append(f"{operator} ({part})")

    # Combine parts with randomly chosen operators
    expression = ' '.join(parts).strip()

    # Final check to make the expression safer for division and ensure it's not empty
    if '/' in expression:
        expression += " + 1"  # Avoid division by zero
    if not expression:
        expression = "0"  # Fallback to a simple zero expression if somehow it ends up empty

    return f"0:({expression})"


class DeforumScheduleTemplate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "expression": (templates,),

            }
               }

    # @classmethod
    # def IS_CHANGED(cls, text, autorefresh):
    #     # Force re-evaluation of the node
    #     if autorefresh == "Yes":
    #         return float("NaN")

    RETURN_TYPES = ("STRING",)
    FUNCTION = "show"
    display_name = "Schedule Templates"
    CATEGORY = "deforum"
    OUTPUT_NODE = True

    def show(self, expression):
        return(str(expression),)

class DeforumAudioScheduleTemplate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "expression": (audio_templates,),

            }
               }

    # @classmethod
    # def IS_CHANGED(cls, text, autorefresh):
    #     # Force re-evaluation of the node
    #     if autorefresh == "Yes":
    #         return float("NaN")

    RETURN_TYPES = ("STRING",)
    FUNCTION = "show"
    display_name = "Audio Schedule Expression Templates"
    CATEGORY = "deforum"
    OUTPUT_NODE = True

    def show(self, expression):
        return(str(expression),)


class DeforumScheduleTemplateRandomizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_frames": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
                "max_parts": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),

            }
               }

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    RETURN_TYPES = ("STRING",)
    FUNCTION = "show"
    display_name = "Schedule Randomizer"
    CATEGORY = "deforum"
    OUTPUT_NODE = True

    def show(self, seed, max_frames, max_parts):
        return(str(generate_complex_random_expression(max_frames, seed, max_parts)),)


class DeforumScheduleVisualizer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "schedule": ("STRING", {"default": "0: (1.0)"}),
                "max_frames": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),

            }
               }

    # @classmethod
    # def IS_CHANGED(cls, text, autorefresh):
    #     # Force re-evaluation of the node
    #     if autorefresh == "Yes":
    #         return float("NaN")

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "show"
    display_name = "Schedule Visualizer"
    CATEGORY = "deforum"
    OUTPUT_NODE = True

    def show(self, schedule, max_frames):

        fi = FrameInterpolator(max_frames, -1)
        series = fi.get_inbetweens(fi.parse_key_frames(schedule))

        # Create a figure for plotting
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        # Plot the series
        ax.plot(series)
        ax.set_title("Schedule Visualization")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")

        # Convert the Matplotlib figure to a PIL Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        pil_img = Image.open(buf)


        return(pil2tensor(pil_img),)