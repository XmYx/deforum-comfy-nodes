import base64
import math
import os
import random
import re
from io import BytesIO

import cv2
import numexpr
import numpy as np
import pandas as pd
import torch
from PIL import Image
from deforum.pipelines.deforum_animation.animation_helpers import DeforumAnimKeys
from deforum.pipelines.deforum_animation.pipeline_deforum_animation import interpolate_areas

blend_methods = ["linear", "sigmoidal", "gaussian", "pyramid", "none"]


def parse_widget(widget_info: dict) -> tuple:
    parsed_widget = None
    t = widget_info["type"]
    if t == "dropdown":
        parsed_widget = (widget_info["choices"],)
    elif t == "checkbox":
        parsed_widget = ("BOOLEAN", {"default": widget_info['default']})
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
    data_info["optional"] = {"deforum_data": ("deforum_data",)}
    return data_info

def get_current_keys(anim_args, seed, root, parseq_args=None, video_args=None, area_prompts=None):
    use_parseq = False if parseq_args == None else True
    anim_args.max_frames += 2
    keys = DeforumAnimKeys(anim_args, seed)  # if not use_parseq else ParseqAnimKeys(parseq_args, video_args)
    areas = None
    # Always enable pseudo-3d with parseq. No need for an extra toggle:
    # Whether it's used or not in practice is defined by the schedules
    if use_parseq:
        anim_args.flip_2d_perspective = True
        # expand prompts out to per-frame
    if use_parseq and keys.manages_prompts():
        prompt_series = keys.prompts
    else:
        prompt_series = None
        if hasattr(root, 'animation_prompts'):
            if root.animation_prompts is not None:
                prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames + 1)])
                for i, prompt in root.animation_prompts.items():
                    if str(i).isdigit():
                        prompt_series[int(i)] = prompt
                    else:
                        prompt_series[int(numexpr.evaluate(i))] = prompt
                prompt_series = prompt_series.ffill().bfill()
        if area_prompts is not None:
            areas = interpolate_areas(area_prompts, anim_args.max_frames)
    anim_args.max_frames -= 2
    return keys, prompt_series, areas

def tensor2pil(image):
    if image is not None:
        #with torch.inference_mode():
        return Image.fromarray(np.clip(255. * image.detach().cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    else:
        return None

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2np(img):
    np_img = np.array(tensor2pil(img))
    return np_img

def get_latent_with_seed(seed):
    return torch.randn(generator=torch.manual_seed(seed))

def pil_image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="WEBP")  # Or JPEG
    return base64.b64encode(buffer.getvalue()).decode()


def tensor_to_webp_base64(tensor):
    # Ensure tensor is in CPU and detach it from the computation graph
    tensor = tensor.clone().detach().cpu()
    # Convert tensor to a numpy array with value range [0, 255]
    # Transpose the tensor to have the channel in the correct order (H, W, C)
    np_image = (tensor.numpy().squeeze() * 255).clip(0, 255).astype(np.uint8)
    if np_image.ndim == 2:  # if it's a grayscale image, convert to RGB
        np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2RGB)
    elif np_image.shape[0] == 3:  # Convert CHW to HWC
        np_image = np_image.transpose(1, 2, 0)
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    # Encode the numpy array to WEBP using OpenCV
    compression_quality = 80
    _, encoded_image = cv2.imencode('.webp', np_image, [cv2.IMWRITE_WEBP_QUALITY, compression_quality])
    # Convert the encoded image to base64
    base64_str = base64.b64encode(encoded_image).decode()
    return base64_str

def generate_seed_list(max_frames, mode='fixed', start_seed=0, step=1):
    """
    Generates a list of seed integers compatible with PyTorch in various manners.

    Parameters:
    - max_frames (int): The maximum number of frames/length of the seed list.
    - mode (str): The mode of seed generation, one of 'fixed', 'random', 'ladder', 'incrementing', or 'decrementing'.
    - start_seed (int): The starting seed value for modes other than 'random'.
    - step (int): The step size for 'incrementing', 'decrementing', and 'ladder' modes.

    Returns:
    - list: A list of seed integers.
    """
    if mode == 'fixed':
        return [start_seed for _ in range(max_frames)]
    elif mode == 'random':
        return [random.randint(0, 2**32 - 1) for _ in range(max_frames)]
    elif mode == 'ladder':
        # Generate a ladder sequence where the sequence is repeated after reaching the max_frames
        return [(start_seed + i // 2 * step if i % 2 == 0 else start_seed + (i // 2 + 1) * step) % (2**32) for i in range(max_frames)]
    elif mode == 'incrementing' or 'iter':
        return [(start_seed + i * step) % (2**32) for i in range(max_frames)]
    elif mode == 'decrementing':
        return [(start_seed - i * step) % (2**32) for i in range(max_frames)]
    else:
        raise ValueError("Invalid mode specified. Choose among 'fixed', 'random', 'ladder', 'incrementing', 'decrementing'.")

def find_next_index(output_dir, filename_prefix, format):
    """
    Finds the next index for an MP4 file given an output directory and a filename prefix.

    Parameters:
    - output_dir: The directory where the MP4 files are saved.
    - filename_prefix: The prefix for the filenames.

    Returns:
    - An integer representing the next index for a new MP4 file.
    """
    # Compile a regular expression pattern to match the filenames
    # This assumes the index is at the end of the filename, before the .mp4 extension
    pattern = re.compile(rf"^{re.escape(filename_prefix)}_(\d+)\.{format.lower()}$")

    max_index = -1
    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            # Extract the current index from the filename
            current_index = int(match.group(1))
            # Update the max index found
            if current_index > max_index:
                max_index = current_index

    # The next index is one more than the highest index found
    next_index = max_index + 1


    return next_index

import torch.nn.functional as F

def pyramid_blend(tensor1, tensor2, blend_value):
    # For simplicity, we'll use two levels of blending
    downsampled1 = F.avg_pool2d(tensor1, 2)
    downsampled2 = F.avg_pool2d(tensor2, 2)

    blended_low = (1 - blend_value) * downsampled1 + blend_value * downsampled2
    blended_high = tensor1 + tensor2 - F.interpolate(blended_low, scale_factor=2)

    return blended_high
def gaussian_blend(tensor2, tensor1, blend_value):
    sigma = 0.5  # Adjust for desired smoothness
    weight = math.exp(-((blend_value - 0.5) ** 2) / (2 * sigma ** 2))
    return (1 - weight) * tensor1 + weight * tensor2
def sigmoidal_blend(tensor1, tensor2, blend_value):
    # Convert blend_value into a tensor with the same shape as tensor1 and tensor2
    blend_tensor = torch.full_like(tensor1, blend_value)
    weight = 1 / (1 + torch.exp(-10 * (blend_tensor - 0.5)))  # Sigmoid function centered at 0.5
    return (1 - weight) * tensor1 + weight * tensor2
def blend_tensors(obj1, obj2, blend_value, blend_method="linear"):
    """
    Blends tensors in two given objects based on a blend value using various blending strategies.
    """

    if blend_method == "linear":
        weight = blend_value
        blended_cond = (1 - weight) * obj1[0] + weight * obj2[0]
        blended_pooled = (1 - weight) * obj1[1]['pooled_output'] + weight * obj2[1]['pooled_output']

    elif blend_method == "sigmoidal":
        blended_cond = sigmoidal_blend(obj1[0], obj2[0], blend_value)
        blended_pooled = sigmoidal_blend(obj1[1]['pooled_output'], obj2[1]['pooled_output'], blend_value)

    elif blend_method == "gaussian":
        blended_cond = gaussian_blend(obj1[0], obj2[0], blend_value)
        blended_pooled = gaussian_blend(obj1[1]['pooled_output'], obj2[1]['pooled_output'], blend_value)

    elif blend_method == "pyramid":
        blended_cond = pyramid_blend(obj1[0], obj2[0], blend_value)
        blended_pooled = pyramid_blend(obj1[1]['pooled_output'], obj2[1]['pooled_output'], blend_value)

    return [[blended_cond, {"pooled_output": blended_pooled}]]
