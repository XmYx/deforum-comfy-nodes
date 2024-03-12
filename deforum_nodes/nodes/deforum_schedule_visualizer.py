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

# audio_templates = [
#     "x * t / max_f",  # Linear scaling of x with respect to time t.
#     "x * sin(2 * pi * t / max_f)",  # Sinusoidal modulation of x based on frame time.
#     "x * (1 - exp(-t / max_f))",  # Exponential approach of x to its maximum over time.
#     "x * exp(-t / max_f)",  # Exponential decay of x over time.
#     "x * sin(2 * pi * e * t / max_f)",  # Sinusoidal modulation with base e to vary frequency.
#     "x * (1 if t < max_f / 2 else 0)",  # Binary step function in time, for a sudden change.
#     "x * log(e + t / max_f)",  # Logarithmic scaling with base e, gentle growth over time.
#     "x * cos(2 * pi * t / max_f) + e ** (t / max_f)",  # Cosine wave plus exponential growth factor e.
#     "x * pow(t / max_f, 2)",  # Quadratic growth of x over time.
#     "x * sqrt(t / max_f)",  # Square root scaling of x, slower increase over time.
#     "x * tan(pi * t / max_f)",  # Tangent modulation, note potential for extreme values.
#     "x * asin(sin(2 * pi * t / max_f))",  # Arcsine of a sinusoidal wave, for harmonic effects.
#     "x * (2 ** (t / max_f) - 1)",  # Exponential growth based on power of 2.
#     "x * factorial(int(t) % 5)",  # Modulating x by the factorial of t modulo 5, for periodic jumps.
#     "x * (1 - abs(2 * t / max_f - 1))",  # Triangle wave shaping of x over time.
#     "x * abs(sin(2 * pi * t / max_f))",  # Absolute value of a sinusoidal wave, ensuring positive values.
#     "x * sin(2 * pi * t / max_f) * cos(2 * pi * t / max_f)",  # Product of sine and cosine for complex modulation.
#     "x * (e ** (cos(2 * pi * t / max_f)) - 1)",  # Exponential function modulated by a cosine wave.
#     "x * if(t < max_f / 2, sin(2 * pi * t / max_f), cos(2 * pi * t / max_f))",  # Conditional modulation with half period sine, half period cosine.
#     "x * sin(4 * pi * t / max_f) + x * cos(2 * pi * t / max_f)",  # Combination of sine and cosine waves with different frequencies.
#     "x * (sin(2 * pi * t / max_f) if t < max_f / 3 else sin(4 * pi * t / max_f))",  # Conditional frequency change in sine wave.
#     "x * (exp(t / max_f) - 1) / log(e + t)",  # Exponential increase tempered by a logarithmic factor.
#     "(x * cos(2 * pi * t / max_f)) / (1 + log(t + 1))",  # Cosine wave scaled by x with a logarithmic denominator to moderate growth.
#     "x * sin(2 * pi * t / max_f) ** 2",  # Square of a sinusoidal wave, creating a smoothed, non-negative waveform.
#     "x * cos(2 * pi * t / max_f) ** 2",  # Square of a cosine wave, similar effect as above but phase-shifted.
# ]


audio_templates = [
    # Simple Amplitude Modulations
    "x * 2",  # Doubling the amplitude.
    "x / 2",  # Halving the amplitude.
    "abs(x)",  # Absolute value of the amplitude.
    "x * x",  # Squaring the amplitude.
    "sqrt(abs(x))",  # Square root of the absolute amplitude.
    "log(abs(x) + 1)",  # Logarithmic scaling of the amplitude.
    "x * sin(x)",  # Sine modulation based on amplitude itself.
    "x * cos(x)",  # Cosine modulation based on amplitude itself.
    "-x",  # Inverting the amplitude.
    "x * (exp(x) - 1)",  # Exponential scaling based on the amplitude.
    "x * pow(e, x - 1)",  # Exponential growth using base e, adjusted for amplitude.
    "x * factorial(int(abs(x)) % 5)",  # Factorial modulation based on the absolute amplitude modulo 5.
    "x if x > 0.5 else x * 2",  # Conditional scaling for amplitudes greater than 0.5.
    "min(x, 0.5)",  # Clipping amplitude to a maximum of 0.5.
    "max(x, -0.5)",  # Ensuring amplitude is not less than -0.5.
    "x % 0.5",  # Amplitude modulo 0.5, creating a repeating pattern.
    "tan(x)",  # Tangent modulation of amplitude.
    "asin(min(1, max(-1, x))) / pi",  # Arcsine of amplitude normalized to [-1, 1], scaled by π.
    "1 / (abs(x) + 1)",  # Inverse scaling of amplitude, avoiding division by zero.
    "pow(e, -abs(x))",  # Exponential decay based on the absolute amplitude.

    # Temporal Expressions Involving Time and Maximum Frame
    "x * t / max_f",  # Linear scaling of amplitude with respect to time.
    "x * sin(2 * pi * t / max_f)",  # Sinusoidal modulation over time.
    "x * exp(-t / max_f)",  # Exponential decay over time.
    "x * (1 - exp(-t / max_f))",  # Inverse exponential growth.
    "x * cos(2 * pi * e * t / max_f)",  # Cosine wave modulation with base e to alter frequency.
    "x * if(t < max_f / 2, 1, 0)",  # Binary switch based on time, for sudden change.
    "x * pow(t / max_f, 2)",  # Quadratic growth of amplitude over time.
    "x * sqrt(t / max_f)",  # Square root scaling over time.
    "x * tan(pi * t / max_f)",  # Tangent modulation over time.
    "x * sin(2 * pi * t / max_f) * cos(2 * pi * t / max_f)",  # Product of sine and cosine for complex temporal modulation.
    "x * sin(2 * pi * t / max_f) if t < max_f / 2 else x * cos(2 * pi * t / max_f)",  # Conditional wave modulation.
    "x * sin(4 * pi * t / max_f) + x * cos(2 * pi * t / max_f)",  # Sum of waves with different frequencies.
    "x * (exp(t / max_f) - 1) / log(e + t)",  # Exponential growth tempered by logarithm of time.
    "(x * cos(2 * pi * t / max_f)) / (1 + log(t + 1))",  # Cosine wave moderated by logarithmic factor.

    # Complex and Combined Expressions
    "x * (sin(2 * pi * t / max_f) ** 2 + cos(2 * pi * t / max_f) ** 2)",  # Sum of squared sine and cosine waves.
    "x * (1 - abs(2 * t / max_f - 1))",  # Triangle wave shaping over time.
    "x * abs(sin(2 * pi * t / max_f))",  # Absolute sine wave for positive values.
    "x * pow(e, cos(2 * pi * t / max_f))",  # Exponential function modulated by cosine.
    "x * asin(sin(2 * pi * t / max_f)) / pi",  # Arcsine of sine wave normalized and scaled.
    "x * pow(2, sin(2 * pi * t / max_f))",  # Exponential growth with base 2, modulated by sine wave.
    "x * log(1 + abs(sin(2 * pi * t / max_f)))",  # Logarithmic scaling modulated by absolute sine wave.
    "x * (sin(2 * pi * t / (max_f + t)))",  # Sinusoidal frequency modulation by time.
    "(x * cos(2 * pi * t / max_f)) / (1 + log(t + 1)) * sin(2 * pi * t / max_f)",  # Combined cosine and sine waves with logarithmic softening.
    "x * (sin(2 * pi * t / max_f) * (t % 10))",  # Sinusoidal wave amplitude modulated by frame modulo.
    "x * exp(t / max_f) * sin(2 * pi * t / max_f)",  # Exponential growth combined with sinusoidal modulation.
    "x * (if(t < max_f / 3, sin(2 * pi * t / max_f), cos(2 * pi * t / max_f)))",  # Conditional frequency modulation.
    "x * (1 / (1 + exp(-t / max_f)))",  # Logistic sigmoid function for smooth transitions.
    "x * pow(e, -t / max_f) * cos(2 * pi * t / max_f)",  # Exponential decay combined with cosine modulation.
    "x * sin(2 * pi * t / max_f) / (1 + exp(-t / max_f))",  # Sinusoidal wave with logistic growth control.
    "x * (e ** (t / max_f) - e ** (-t / max_f)) / 2",  # Hyperbolic sine function for symmetric exponential growth and decay.
    "x * (1 - (t / max_f) ** 2)",  # Parabolic decrease towards the end of the timeline.
    "x * tanh(2 * pi * t / max_f)",  # Hyperbolic tangent for smooth transitions between -1 and 1.
    "x * factorial((int(t) % 5) + 1)",  # Factorial modulation based on time modulo 5.
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
    CATEGORY = "deforum/help"
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
    CATEGORY = "deforum/help"
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
    CATEGORY = "deforum/utils"
    OUTPUT_NODE = True

    def show(self, seed, max_frames, max_parts):
        return(str(generate_complex_random_expression(max_frames, seed, max_parts)),)


class DeforumScheduleVisualizer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "schedule": ("STRING", {"default": "0: (1.0)"}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 128000, "step": 1}),
                "grid": ("BOOLEAN", {"default": False}),

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
    CATEGORY = "deforum/utils"
    OUTPUT_NODE = True

    def show(self, schedule, max_frames, grid):
        if max_frames == 0:
            max_frames = len(schedule.split(','))
        fi = FrameInterpolator(max_frames, -1)
        series = fi.get_inbetweens(fi.parse_key_frames(schedule))

        # Create a figure for plotting
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        if grid:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

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