import secrets

import torch
import numpy as np
import math

import pywt
import torch.nn.functional as F
from opensimplex import OpenSimplex



class AddAdvancedNoiseNode:
    """
    A node to add various types of advanced noise to an image using PyTorch.
    """
    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "noise_type": (["wavelet", "value", "flow", "turbulence",
                                "ridged_multifractal", "reaction_diffusion", "voronoi", "simplex"],), # "gabor"
                "amount": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": None,
                    "min": 0,
                    "max": 2 ** 32 - 1,
                    "step": 1,
                    "display": "number"
                }),
                # Voronoi specific parameters
                "num_points": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
                # Simplex Noise specific parameters
                "scale": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "octaves": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
                "persistence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "lacunarity": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1,
                    "display": "number"
                }),
                # Wavelet Noise specific parameters

                "wavelet": (["haar", "db1", "sym2", "coif1", "bior1.3", "rbio1.3"],),
                "mode": (["symmetric", "periodic", "reflect", "zero-padding"],),
                # Additional parameters specific to Gabor noise
                "frequency": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "theta": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2 * math.pi,
                    "step": 0.01,
                    "display": "slider"
                }),
                "sigma_x": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "sigma_y": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 50.0,
                    "step": 1.0,
                    "display": "number"
                }),
                # Value Noise specific parameters
                "res": ("INT", {
                    "default": 16,
                    "min": 4,
                    "max": 64,
                    "step": 1,
                    "display": "number"
                }),
                # Additional parameters for Flow Noise
                "flow_scale": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "flow_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2 * np.pi,
                    "step": 0.1,
                    "display": "number"
                }),
                # Reaction Diffusion Noise
                "steps": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "Du": ("FLOAT", {
                    "default": 0.16,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number"
                }),
                "Dv": ("FLOAT", {
                    "default": 0.08,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "number"
                }),
                "feed_rate": ("FLOAT", {
                    "default": 0.035,
                    "min": 0.01,
                    "max": 0.1,
                    "step": 0.001,
                    "display": "slider"
                }),
                "kill_rate": ("FLOAT", {
                    "default": 0.06,
                    "min": 0.01,
                    "max": 0.1,
                    "step": 0.001,
                    "display": "slider"
                }),
                # Additional parameters specific to each noise type can be defined here
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("NOISED_IMAGE","NOISE",)

    FUNCTION = "add_advanced_noise"
    CATEGORY = "Advanced Noise"
    display_name = "Add Advanced Noise"

    def add_advanced_noise(self, images, noise_type, amount, seed=None, **kwargs):
        B, H, W, C = images.shape
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Adjusted noise function dictionary
        noise_function = {
            "voronoi": generate_voronoi_noise,
            "simplex": lambda H, W, **kw: generate_simplex_noise(H, W, **kw).unsqueeze(-1).repeat(1, 1, 1, C),
            "wavelet": lambda H, W, **kw: generate_wavelet_noise(H, W, **kw).unsqueeze(-1).repeat(1, 1, 1, C),
            "gabor": lambda H, W, **kw: generate_gabor_noise(H, W, **kw),
            "value": lambda H, W, **kw: generate_value_noise(H, W, **kw).unsqueeze(-1).repeat(1, 1, 1, C),
            "flow": lambda H, W, **kw: generate_flow_noise(H, W, **kw),
            "turbulence": lambda H, W, **kw: generate_turbulence_noise(H, W, **kw),
            "ridged_multifractal": lambda H, W, **kw: generate_ridged_multifractal_noise(H, W, **kw),
            "reaction_diffusion": lambda H, W, **kw: generate_reaction_diffusion_noise(H, W, **kw),

            # Additional noise functions adjusted similarly
        }.get(noise_type, lambda H, W, **kw: torch.zeros((B, H, W, C)))

        # Generate the noise pattern
        noise = noise_function(H, W, seed=seed, **kwargs)
        # Adjust for the case where noise is not returned with a batch dimension or channel dimension
        if noise.dim() == 2:  # Assuming noise is of shape (H, W)
            noise = noise.unsqueeze(0).unsqueeze(-1)  # Add batch and channel dimension
            noise = noise.expand(B, -1, -1, C)  # Expand to match (B, H, W, C)
        elif noise.dim() == 3:  # Assuming noise is of shape (B, H, W)
            noise = noise.unsqueeze(-1)  # Add channel dimension
            noise = noise.expand(-1, -1, -1, C)  # Expand to match (B, H, W, C)

        # Ensure noise tensor device matches the images tensor device
        noise = noise.to(images.device)
        # from ..modules.better_resize.resize_right import resize
        # belnd_img = resize(images, noise.shape, antialiasing=False)
        # Apply the noise to the images
        noisy_images = images.clone() + amount * noise.clone()

        return (torch.clamp(noisy_images, 0, 1),noise,)

def generate_voronoi_noise(H, W, num_points=100, seed=None, *args, **kwargs):
    """
    Generates a 2D Voronoi (Worley) Noise pattern.

    Args:
        H, W (int): The height and width of the output noise pattern.
        num_points (int): Number of seed points to generate.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: A tensor containing the Voronoi noise pattern.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Generate random seed points
    seeds_x = np.random.uniform(0, W, num_points)
    seeds_y = np.random.uniform(0, H, num_points)
    seeds = np.stack((seeds_x, seeds_y), axis=-1)

    # Initialize the noise tensor
    noise = torch.full((H, W), float('inf'))

    # Compute distance from each pixel to the closest seed point
    for y in range(H):
        for x in range(W):
            distances = torch.tensor([((x - sx) ** 2 + (y - sy) ** 2) for sx, sy in seeds])
            noise[y, x] = torch.min(distances).sqrt()

    # Normalize the noise pattern
    noise = (noise - noise.min()) / (noise.max() - noise.min())

    return noise

def generate_simplex_noise(H, W, scale=0.05, octaves=4, persistence=0.5, lacunarity=2.0, seed=None, **kwargs):
    if seed is not None:
        gen = OpenSimplex(seed)
    else:
        gen = OpenSimplex()

    noise_pattern = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            nx = x / W - 0.5  # Normalize x coordinate to [-0.5, 0.5]
            ny = y / H - 0.5  # Normalize y coordinate to [-0.5, 0.5]
            noise_val = 0
            amplitude = 1.0
            frequency = 1.0
            max_value = 0.0
            for o in range(octaves):
                noise_val += gen.noise2(x=nx * scale * frequency, y=ny * scale * frequency) * amplitude
                max_value += amplitude
                amplitude *= persistence
                frequency *= lacunarity

            # Normalize each point individually
            noise_pattern[y, x] = (noise_val / max_value + 1) / 2  # Normalize to [0, 1]

    return torch.from_numpy(noise_pattern)


def generate_wavelet_noise(H, W, octaves=3, wavelet='haar', mode='symmetric', **kwargs):
    seed = kwargs.get('seed', secrets.randbelow(2 ** 32))
    if seed is not None:
        np.random.seed(seed)

    # Generate initial random noise
    noise = np.random.randn(H, W)

    # Perform wavelet decomposition and reconstruction
    coeffs = pywt.wavedec2(noise, wavelet=wavelet, mode=mode, level=octaves)
    wavelet_noise = pywt.waverec2(coeffs, wavelet=wavelet, mode=mode)

    # Normalize the wavelet noise to have values between 0 and 1
    wavelet_noise = (wavelet_noise - np.min(wavelet_noise)) / (np.max(wavelet_noise) - np.min(wavelet_noise))

    # Convert to a PyTorch tensor and add a channel dimension since input is assumed to be BxHxWxC
    wavelet_noise_tensor = torch.from_numpy(wavelet_noise).float().unsqueeze(0).unsqueeze(0)

    # Interpolate to match input size (H, W)
    # Note: 'align_corners=False' is generally used for non-exact sizes but you might want to experiment
    # with 'align_corners=True' or 'recompute_scale_factor=True' depending on your specific use case
    wavelet_noise_tensor = F.interpolate(wavelet_noise_tensor, size=(H, W), mode='bilinear', align_corners=False)

    # Remove the extra dimensions added for interpolation if necessary
    wavelet_noise_tensor = wavelet_noise_tensor.squeeze()

    return wavelet_noise_tensor


def generate_gabor_kernel(frequency, theta, sigma_x, sigma_y, n_stds=3, grid_size=28):
    """
    Generates a Gabor kernel.

    Args:
        frequency (float): The frequency of the sinusoidal component.
        theta (float): Orientation of the Gabor filter in radians.
        sigma_x, sigma_y (float): Standard deviation of the Gaussian envelope along x and y axes.
        n_stds (int): Number of standard deviations to consider for the kernel size.
        grid_size (int): The size of the output tensor.

    Returns:
        torch.Tensor: A 2D tensor with the Gabor kernel.
    """
    xmax = ymax = grid_size // 2
    xmin = ymin = -xmax
    (y, x) = torch.meshgrid(torch.arange(ymin, ymax + 1), torch.arange(xmin, xmax + 1))
    x = x.float()
    y = y.float()

    # Convert theta to a tensor to use with torch.cos and torch.sin
    theta_tensor = torch.tensor(theta)

    rotx = x * torch.cos(theta_tensor) + y * torch.sin(theta_tensor)
    roty = -x * torch.sin(theta_tensor) + y * torch.cos(theta_tensor)

    g = torch.zeros(y.shape)
    g = torch.exp(-0.5 * ((rotx ** 2 / sigma_x ** 2) + (roty ** 2 / sigma_y ** 2))) * torch.cos(
        2 * np.pi * frequency * rotx)

    return g

def generate_gabor_noise(H, W, frequency=0.1, theta=0.0, sigma_x=10.0, sigma_y=10.0, batch_size=1, channels=3, **kwargs):
    """
    Generates a 2D Gabor Noise pattern and scales it to the size of the input images.

    Args:
        H, W (int): Height and width of the output noise pattern.
        frequency (float): Frequency of the sinusoidal component.
        theta (float): Orientation of the Gabor filter in radians.
        sigma_x, sigma_y (float): Standard deviation of the Gaussian envelope.
        batch_size (int): Number of images in the batch.
        channels (int): Number of channels in the images.

    Returns:
        torch.Tensor: A tensor containing the Gabor noise pattern, resized to match input images dimensions.
    """
    gabor_kernel = generate_gabor_kernel(frequency, theta, sigma_x, sigma_y, grid_size=min(H, W))

    # Center the kernel in a larger tensor matching the image dimensions
    pad_height = (H - gabor_kernel.size(0)) // 2
    pad_width = (W - gabor_kernel.size(1)) // 2
    pad_height_extra = H - gabor_kernel.size(0) - pad_height * 2
    pad_width_extra = W - gabor_kernel.size(1) - pad_width * 2

    # Pad the kernel to match the target size
    gabor_padded = F.pad(gabor_kernel, [pad_width, pad_width + pad_width_extra, pad_height, pad_height_extra], "constant", 0)

    # Ensure the pattern has the correct dimensions: batch_size x H x W x channels
    gabor_padded = gabor_padded.unsqueeze(0)  # Add a batch dimension
    gabor_padded = gabor_padded.unsqueeze(-1)  # Add a channel dimension
    gabor_padded = gabor_padded.expand(batch_size, -1, -1, channels)  # Expand to match the full dimensions

    return gabor_padded


def generate_value_noise(H, W, res=16, seed=None, **kwargs):
    """
    Generates a 2D Value Noise pattern with edge handling.

    Args:
        H, W (int): Height and width of the output noise pattern.
        res (int): Resolution of the noise grid (smaller values mean smoother noise).
        seed (int, optional): Seed for random number generator.

    Returns:
        torch.Tensor: A tensor containing the Value noise pattern.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Create a grid of random values
    grid_res_x, grid_res_y = W // res + 1, H // res + 1
    random_values = torch.rand((grid_res_y, grid_res_x))

    # Initialize output noise pattern
    noise_pattern = torch.zeros((H, W))

    # Generate the noise
    for y in range(H):
        for x in range(W):
            # Grid cell coordinates in the random grid
            cell_x, cell_y = x // res, y // res

            # Ensure indices stay within bounds
            next_cell_x = min(cell_x + 1, grid_res_x - 1)
            next_cell_y = min(cell_y + 1, grid_res_y - 1)

            # Local x and y in the grid cell
            local_x, local_y = (x % res) / res, (y % res) / res

            # Corners of the cell in the grid
            c00 = random_values[cell_y, cell_x]
            c10 = random_values[cell_y, next_cell_x]
            c01 = random_values[next_cell_y, cell_x]
            c11 = random_values[next_cell_y, next_cell_x]

            # Interpolate between grid corner values
            nx0 = lerp(c00, c10, fade(local_x))
            nx1 = lerp(c01, c11, fade(local_x))
            nxy = lerp(nx0, nx1, fade(local_y))

            noise_pattern[y, x] = nxy

    # Normalize the noise pattern
    noise_pattern = (noise_pattern - noise_pattern.min()) / (noise_pattern.max() - noise_pattern.min())

    return noise_pattern

def lerp(a, b, t):
    """Linear interpolation between a and b with t in [0, 1]."""
    return a + t * (b - a)

def fade(t):
    """Fade function as defined by Ken Perlin; eases coordinate values."""
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_flow_noise_pattern(H, W, scale=0.1, angle=0.0, seed=None):
    """
    Generates a 2D Flow Noise pattern using a base noise to perturb the sampling space of another noise layer.

    Args:
        H, W (int): Height and width of the output noise pattern.
        scale (float): Scale of the underlying noise.
        angle (float): Base flow direction in radians.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.Tensor: A tensor containing the Flow noise pattern.
    """
    # Base noise for perturbation
    base_noise = generate_simplex_noise(H, W, scale=scale, seed=seed)
    # Generate gradient from angle for flow direction
    flow_x = torch.cos(torch.tensor(angle))
    flow_y = torch.sin(torch.tensor(angle))

    # Apply perturbation based on the base noise and flow direction
    perturb_x = base_noise * flow_x
    perturb_y = base_noise * flow_y

    # Another layer of noise for actual flow appearance
    flow_noise = generate_simplex_noise(H, W, scale=scale / 2, seed=seed if seed is None else seed + 1)

    # Final flow noise is a combination of perturbed coordinates and the flow noise layer
    final_noise = flow_noise + perturb_x + perturb_y

    # Normalize the noise pattern
    final_noise = (final_noise - final_noise.min()) / (final_noise.max() - final_noise.min())

    return final_noise


def generate_flow_noise(H, W, flow_scale=0.1, flow_angle=0.0, seed=None, batch_size=1, channels=3, **kwargs):
    """
    Generates a 2D Flow Noise pattern and resizes it to match the input image size.

    Args:
        H, W (int): Height and width of the input images.
        scale, angle: Parameters for the flow noise generation.
        seed (int, optional): Seed for random number generator.
        batch_size (int): The batch size of the input images.
        channels (int): The channel count of the input images.

    Returns:
        torch.Tensor: A tensor of the Flow noise pattern, matched to input dimensions.
    """
    # Generate the base noise pattern
    base_noise = generate_flow_noise_pattern(H, W, flow_scale, flow_angle, seed)
    # Ensure the noise tensor has the same dimensions as the input: (batch_size, H, W, channels)
    # Interpolate noise to match the input size if needed
    noise_resized = F.interpolate(base_noise.unsqueeze(0).unsqueeze(0), size=(H, W), mode='bilinear',
                                  align_corners=False)

    # Expand to match the batch and channel dimensions
    noise_resized = noise_resized.repeat(batch_size, channels, 1, 1)
    noise_resized = noise_resized.permute(0, 2, 3, 1)  # Rearrange dimensions to match (B, H, W, C)

    return noise_resized

def generate_turbulence_noise(H, W, scale=0.05, octaves=4, persistence=0.5, lacunarity=2.0, seed=None, **kwargs):
    if seed is not None:
        gen = OpenSimplex(seed)
    else:
        gen = OpenSimplex()

    noise_pattern = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            nx = x / W - 0.5  # Normalize x coordinate to [-0.5, 0.5]
            ny = y / H - 0.5  # Normalize y coordinate to [-0.5, 0.5]
            noise_val = 0
            amplitude = 1.0
            frequency = 1.0
            max_value = 0.0
            for o in range(octaves):
                # The key change for turbulence: using the absolute value of the noise
                noise_val += abs(gen.noise2(x=nx * scale * frequency, y=ny * scale * frequency)) * amplitude
                max_value += amplitude
                amplitude *= persistence
                frequency *= lacunarity

            # Normalize each point individually
            noise_pattern[y, x] = (noise_val / max_value)

    # Normalize the whole pattern to [0, 1] range if desired
    noise_pattern = (noise_pattern - noise_pattern.min()) / (noise_pattern.max() - noise_pattern.min())

    return torch.from_numpy(noise_pattern)

def generate_ridged_multifractal_noise(H, W, scale=0.05, octaves=4, persistence=0.5, lacunarity=2.0, seed=None, **kwargs):
    if seed is not None:
        gen = OpenSimplex(seed)
    else:
        gen = OpenSimplex()

    noise_pattern = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            nx = x / W - 0.5  # Normalize x coordinate to [-0.5, 0.5]
            ny = y / H - 0.5  # Normalize y coordinate to [-0.5, 0.5]
            noise_val = 0
            amplitude = 1.0
            frequency = 1.0
            max_value = 0.0
            for o in range(octaves):
                signal = gen.noise2(x=nx * scale * frequency, y=ny * scale * frequency)
                # Transformation for ridged noise: invert and square the signal
                signal = 1.0 - abs(signal)
                signal *= signal
                noise_val += signal * amplitude
                max_value += amplitude
                amplitude *= persistence
                frequency *= lacunarity

            # Normalize each point individually
            noise_pattern[y, x] = (noise_val / max_value)

    # Normalize the whole pattern to [0, 1] range if desired
    noise_pattern = (noise_pattern - noise_pattern.min()) / (noise_pattern.max() - noise_pattern.min())

    return torch.from_numpy(noise_pattern)


def generate_reaction_diffusion_noise(H, W, steps=100, Du=0.16, Dv=0.08, feed_rate=0.035, kill_rate=0.06, seed=None, **kwargs):
    """
    Generates a 2D Reaction-Diffusion pattern using a simplified Gray-Scott model.

    Args:
        H, W (int): Height and width of the output noise pattern.
        steps (int): Number of simulation steps.
        Du, Dv (float): Diffusion rates for substances A and B.
        feed_rate, kill_rate (float): Rates for feed and kill reactions.
        seed (int, optional): Seed for initializing the pattern.

    Returns:
        torch.Tensor: A tensor containing the Reaction-Diffusion noise pattern.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Initialize A and B concentrations with A being fully saturated and B having a noise pattern
    A = torch.ones((H, W), dtype=torch.float32)
    B = torch.rand((H, W), dtype=torch.float32) * 0.25  # Starting with a low concentration of B

    # Laplacian kernel for diffusion
    laplacian_kernel = torch.tensor([[0.05, 0.2, 0.05],
                                     [0.2, -1.0, 0.2],
                                     [0.05, 0.2, 0.05]], dtype=torch.float32)

    for _ in range(steps):
        LA = torch.nn.functional.conv2d(A.unsqueeze(0).unsqueeze(0), laplacian_kernel.unsqueeze(0).unsqueeze(0), padding='same').squeeze()
        LB = torch.nn.functional.conv2d(B.unsqueeze(0).unsqueeze(0), laplacian_kernel.unsqueeze(0).unsqueeze(0), padding='same').squeeze()

        # Reaction-diffusion equations
        AB2 = A * B.pow(2)
        dA = Du * LA - AB2 + feed_rate * (1 - A)
        dB = Dv * LB + AB2 - (feed_rate + kill_rate) * B

        A += dA
        B += dB

    # Normalize the B concentration pattern to be between 0 and 1
    pattern = (B - B.min()) / (B.max() - B.min())

    return pattern.unsqueeze(0)  # Add a channel dimension
