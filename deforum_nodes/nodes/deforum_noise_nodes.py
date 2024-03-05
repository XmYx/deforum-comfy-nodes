import secrets

import torch
import numpy as np


class AddCustomNoiseNode:
    """
    A node to add various types of noise to an image using PyTorch.
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
                "noise_type": (
                ["speckle", "uniform", "rayleigh", "exponential",
                 "gamma", "random_valued_impulse", "laplace", "perlin", "brownian", "quantization", "shot",
                 "multiplicative", "flicker", "fractal", "cellular", "gaussian", "thermal", "salt_pepper", "poisson",],), #TODO ["blue", "anisotropic"]
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
                "temperature_map": ("IMAGE",),
                "mean": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "std": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "prob": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "scale": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "temp_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "scale_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "location": ("FLOAT", {
                    "default": 0.0,
                    "min": -10.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "res_x": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "res_y": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "octaves": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
                "persistence": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
                "num_points": ("INT", {
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "display": "number"
                }),
                "direction": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.00,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("NOISED_IMAGE","NOISE",)
    FUNCTION = "add_noise"
    display_name = "Add Custom Noise"
    CATEGORY = "deforum"
    def add_noise(self, images, noise_type, amount, seed=None, temperature_map=None, **kwargs):

        image, noise = add_noise_torch(images.clone(), noise_type, seed, amount, temperature_map, **kwargs)
        return (image,noise,)



def add_noise_torch(images, noise_type='gaussian', seed=None, amount=0.1, temperature_map=None, **kwargs):
    """
    Add various types of noise to an image tensor using PyTorch.

    Parameters:
        images (torch.Tensor): The input images tensor of shape (B, C, H, W).
        noise_type (str): Type of noise to add. Supports 'gaussian', 'salt_pepper', 'poisson', 'speckle',
                          'uniform', 'rayleigh', 'exponential', 'gamma', 'random_valued_impulse'.
        seed (int): Seed value for randomness. Default is None.
        amount (float): General parameter to control noise amount, interpretation depends on noise_type.
        **kwargs: Additional noise-specific parameters.

    Returns:
        torch.Tensor: The noisy images tensor of the same shape as input.
    """
    B, C, H, W = images.shape

    noise = images

    if seed is not None:
        torch.manual_seed(seed)
    else:
        seed = secrets.randbelow(2 ** 32)
    if noise_type == 'gaussian':
        mean = kwargs.get('mean', 0.0)
        std = kwargs.get('std', 0.1)
        noisy_images = images + amount * torch.randn_like(images) * std + mean

    elif noise_type == 'thermal':
        if temperature_map is None:
            raise ValueError("Temperature map must be provided for thermal noise type.")
        # Normalize temperature map to have a meaningful scale for noise
        temp_min = temperature_map.min()
        temp_max = temperature_map.max()
        normalized_temp_map = (temperature_map - temp_min) / (temp_max - temp_min)
        # Scale temperature map to control the amount of thermal noise
        temp_scale = kwargs.get('temp_scale', 1.0)  # Scale factor for temperature effect
        std_map = amount * normalized_temp_map * temp_scale
        # Generate thermal noise based on temperature map
        noise = torch.randn_like(images) * std_map.unsqueeze(
            1)  # Ensure std_map matches the channel dimension of images
        noisy_images = images + noise

    elif noise_type == 'salt_pepper':
        prob = kwargs.get('prob', 0.05)  # Probability for both salt and pepper
        mask = torch.rand_like(images) < prob
        images[mask] = torch.rand(1).item() * mask[mask]  # Salt
        mask = torch.rand_like(images) < prob
        images[mask] = 0  # Pepper
        noisy_images = images
    elif noise_type == 'poisson':
        noisy_images = torch.poisson(images * amount) / amount

    elif noise_type == 'speckle':
        noise = torch.randn_like(images)
        noisy_images = images + images * noise * amount

    elif noise_type == 'uniform':
        noise = torch.rand_like(images)
        noisy_images = images + amount * (noise - 0.5)

    elif noise_type == 'rayleigh':
        scale = kwargs.get('scale', 0.1)
        noise = torch.sqrt(-2 * scale * torch.log(1 - torch.rand_like(images)))
        noisy_images = images + amount * noise

    elif noise_type == 'exponential':
        scale = kwargs.get('scale', 0.1)
        noise = torch.distributions.Exponential(scale).sample(images.shape)
        noisy_images = images + amount * noise
        noise = torch.clamp(1 / noise, 0, 1.0) * amount
    elif noise_type == 'gamma':
        shape = kwargs.get('shape', 2.0)
        scale = kwargs.get('scale', 0.1)
        noise = torch.distributions.Gamma(shape, scale).sample(images.shape)
        noisy_images = images + amount * noise
        noise = torch.clamp(1 / noise, 0, 1.0) * amount

    elif noise_type == 'random_valued_impulse':
        prob = kwargs.get('prob', 0.05)
        mask = torch.rand_like(images) < prob
        # Fix: Use logical NOT operator on the mask
        inverted_mask = ~mask
        noise = torch.rand_like(images) * mask
        noisy_images = images * inverted_mask.float() + noise
        noise = (torch.clamp(noise * 255.0, 0, 1.0) * mask) * amount
    elif noise_type == 'laplace':
        location = kwargs.get('location', 0.0)
        scale = kwargs.get('scale', 0.1)
        noise = torch.distributions.Laplace(location, scale).sample(images.shape)
        noisy_images = images + amount * noise
        noise = torch.clamp(noise * 255.0, 0, 1.0) * amount

    elif noise_type == 'perlin':
        # Hypothetical function call - you need to implement or integrate a Perlin noise generator.
        noise = generate_perlin_noise(B, C, H, W, **kwargs)
        noisy_images = images + amount * noise

    elif noise_type == 'brownian':
        scale = kwargs.get('scale', 0.1)
        noise = generate_brownian_noise(images.shape, scale, seed)
        noisy_images = images + amount * noise / torch.max(noise)
    elif noise_type == 'quantization':
        # Quantization levels
        levels = kwargs.get('levels', 256)  # Default is 256 levels for an 8-bit image
        # Quantize the image
        max_val = images.max()
        quantized_images = torch.round(images * (levels - 1) / max_val) * max_val / (levels - 1)
        # Calculate quantization noise as the difference between original and quantized images
        noise = images - quantized_images
        # Optionally scale the noise
        noisy_images = images + amount * noise
        noise = torch.clamp(noise, 0, 1.0)
    elif noise_type == 'shot':
        # Scale images to simulate intensity as photon counts
        scale_factor = kwargs.get('scale_factor', 1.0)  # Scale factor to adjust the intensity
        scaled_images = images * scale_factor

        # Apply Poisson distribution to simulate shot noise
        # Since Poisson function expects lambda > 0, ensure scaled_images is positive
        noisy_images = torch.poisson(scaled_images) / scale_factor
    elif noise_type == 'multiplicative':
        # Amount parameter controls the variance of the multiplicative noise
        mean = kwargs.get('mean', 1.0)  # Multiplicative noise mean. Typically, it should be around 1.
        std = kwargs.get('std', 0.1)  # Standard deviation for the multiplicative noise

        # Generate multiplicative noise
        noise = torch.randn_like(images) * std + mean

        # Apply the multiplicative noise
        noisy_images = images * noise
    elif noise_type == 'blue':
        # Generate approximate Blue Noise
        noise = generate_approx_blue_noise((B, C, H, W), seed=seed, **kwargs)

        # Apply the Blue Noise pattern to the images
        # The amount parameter controls the visibility of the blue noise
        noisy_images = images + amount * noise
    elif noise_type == 'flicker':
        # Create frequency domain noise
        # Create frequency domain noise
        real_part = torch.randn((B, C, H, W // 2 + 1), device=images.device, generator=torch.manual_seed(seed))
        imag_part = torch.randn((B, C, H, W // 2 + 1), device=images.device, generator=torch.manual_seed(seed))
        freq_domain_noise = real_part + 1j * imag_part

        # Correctly compute frequencies for 1/f scaling
        frequencies_x = torch.fft.rfftfreq(W, d=1 / W).to(images.device)
        frequencies_y = torch.fft.fftfreq(H, d=1 / H).to(images.device)

        # Use broadcasting to create a meshgrid of frequencies
        freq_mesh_x = frequencies_x[None, None, None, :]  # Add dimensions for B, C, and H for broadcasting
        freq_mesh_y = frequencies_y[None, None, :, None]  # Add dimensions for B, C, and W for broadcasting

        # Calculate the hypotenuse of the frequency meshgrid to apply 1/f scaling
        freq_mesh = torch.sqrt(freq_mesh_x ** 2 + freq_mesh_y ** 2)

        # Avoid division by zero by adding a small value, no need to expand as broadcasting handles it
        freq_domain_noise /= (freq_mesh + 1e-5)

        # Convert back to the spatial domain
        noise = torch.fft.irfft(freq_domain_noise, n=W, dim=-1)

        # Apply noise to images
        noisy_images = images + amount * noise
    elif noise_type == 'anisotropic':
        # Generate base noise
        base_noise = torch.randn_like(images)
        # Apply a directional gradient
        dir_min = kwargs.get('dir_min', 0.0)
        dir_max = kwargs.get('dir_min', 1.0)
        direction = torch.tensor([dir_max, dir_min])
        gradient = torch.outer(torch.arange(H), direction[0]) + torch.outer(torch.arange(W), direction[1])
        gradient = gradient.unsqueeze(0).unsqueeze(0).to(images.device)
        # Apply directional bias to the noise
        noise = base_noise * gradient
        noisy_images = images + amount * noise
    elif noise_type == 'fractal':
        res_x = kwargs.get('res_x', 10)  # Base resolution for Perlin noise
        res_y = kwargs.get('res_y', 10)
        octaves = kwargs.get('octaves', 5)  # Number of noise layers
        persistence = kwargs.get('persistence', 0.5)  # Amplitude decay per layer
        noise = generate_fractal_noise(B, C, H, W, (res_x, res_y), octaves, persistence)
        noisy_images = images + amount * noise
    elif noise_type == 'cellular':
        num_points = kwargs.get('num_points', 100)  # Default number of points
        cellular_noise = generate_cellular_noise(H, W, num_points=num_points)
        # Expand Cellular noise to match input dimensions
        noise = cellular_noise.unsqueeze(0).unsqueeze(0).repeat(B, C, 1, 1)
        noisy_images = images + amount * noise
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

    return torch.clamp(noisy_images, 0, 1), noise


def generate_perlin_noise_2d(shape, res, seed=None):
    """
    Generate a 2D numpy array of Perlin noise.

    Args:
        shape: The shape of the generated array (height, width).
        res: The resolution of the noise in terms of number of cells in each dimension.

    Returns:
        A 2D numpy array of Perlin noise.
    """

    if seed is not None:
        np.random.seed(seed)  # Set the seed if provided

    def interpolate(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def gradient(h, x, y):
        vectors = np.array([[0,1],[0,-1],[1,0],[-1,0]])
        g = vectors[h % 4]
        return g[:, :, 0] * x + g[:, :, 1] * y

    grid_x, grid_y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # Rescale grid to fit res
    grid_x = grid_x * res[0] / shape[1]
    grid_y = grid_y * res[1] / shape[0]

    # Determine grid cell coordinates
    x0 = grid_x.astype(int)
    y0 = grid_y.astype(int)

    # Relative x and y coordinates within each cell
    x_rel = grid_x - x0
    y_rel = grid_y - y0

    # Random gradients
    np.random.seed(0)  # Optional: for reproducibility
    gradients = np.random.randint(0, 4, (res[0] + 1, res[1] + 1))

    # Compute the dot-product
    n0 = gradient(gradients[y0, x0], x_rel, y_rel)
    n1 = gradient(gradients[y0, x0 + 1], x_rel - 1, y_rel)
    ix0 = interpolate(n0) + (interpolate(n1) - interpolate(n0)) * interpolate(x_rel)

    n0 = gradient(gradients[y0 + 1, x0], x_rel, y_rel - 1)
    n1 = gradient(gradients[y0 + 1, x0 + 1], x_rel - 1, y_rel - 1)
    ix1 = interpolate(n0) + (interpolate(n1) - interpolate(n0)) * interpolate(x_rel)

    value = ix0 + (ix1 - ix0) * interpolate(y_rel)
    return (value - np.min(value)) / (np.max(value) - np.min(value))

def generate_perlin_noise(B, C, H, W, res_x, res_y, seed=None, *args, **kwargs):
    """
    Generates Perlin noise for a batch of images in PyTorch.

    Args:
        B, C, H, W: Dimensions of the output tensor.
        res_x, res_y: Resolution of the Perlin noise grid.

    Returns:
        A PyTorch tensor of shape (B, C, H, W) containing Perlin noise.
    """
    noise = np.zeros((B, C, H, W))
    for b in range(B):
        for c in range(C):
            noise[b, c] = generate_perlin_noise_2d((H, W), (res_x, res_y), seed=seed)
    return torch.FloatTensor(noise)


def generate_brownian_noise(shape, scale=1.0, seed=None):
    """
    Generate Brownian (Red) noise for a given tensor shape.

    Parameters:
    - shape (tuple): The shape of the output tensor (e.g., (B, C, H, W)).
    - scale (float): Scaling factor for noise intensity. Default is 1.0.
    - seed (int): Optional seed for reproducibility. Default is None.

    Returns:
    - torch.Tensor: Tensor containing Brownian noise of the specified shape.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Initialize a tensor of random noise
    random_noise = torch.randn(shape)

    # Cumulatively sum the noise along the last dimension to simulate Brownian motion
    brownian_noise = torch.cumsum(random_noise, dim=-1) * scale

    # Normalize the noise to have values between 0 and 1
    brownian_noise -= brownian_noise.min()
    brownian_noise /= brownian_noise.max()

    return brownian_noise


def generate_approx_blue_noise(shape, seed=None, min_dist=1.0, sample_fraction=0.1, *args, **kwargs):
    """
    Generates an approximate Blue Noise pattern using a simplified, naive approach.
    Note: This is computationally expensive and not optimized for performance or quality.

    Parameters:
        shape (tuple): Shape of the output tensor (B, C, H, W).
        seed (int): Random seed for reproducibility.
        min_dist (float): Minimum distance between points. Controls the density.
        sample_fraction (float): Fraction of points to sample, lower means sparser.

    Returns:
        torch.Tensor: Tensor containing an approximate Blue Noise pattern.
    """
    if seed is not None:
        torch.manual_seed(seed)

    _, _, H, W = shape
    pattern = torch.zeros((H, W))

    # Number of points to sample
    num_points = int(H * W * sample_fraction)

    # Randomly sample points with minimum distance
    for _ in range(num_points):
        x, y = torch.randint(0, H, (1,)), torch.randint(0, W, (1,))

        # Check if any existing point is within min_dist
        y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        distances = torch.sqrt((x_grid - x) ** 2 + (y_grid - y) ** 2)
        if torch.min(distances * (pattern > 0).float()) >= min_dist or torch.sum(pattern) == 0:
            pattern[x, y] = 1

    # Convert the pattern to match the input shape, assuming binary pattern applied across all channels
    blue_noise_pattern = pattern.repeat(shape[1], 1, 1).unsqueeze(0).repeat(shape[0], 1, 1, 1)

    return blue_noise_pattern.float()


def generate_fractal_noise(B, C, H, W, res, octaves=5, persistence=0.5, *args, **kwargs):
    """
    Generates fractal noise for a batch of images in PyTorch.

    Args:
        B (int): Batch size.
        C (int): Number of channels.
        H, W (int): Height and Width of the images.
        res (tuple): Resolution of the base Perlin noise (res_x, res_y).
        octaves (int): Number of layers of noise to generate.
        persistence (float): Amplitude decay per octave (determines "roughness").

    Returns:
        A PyTorch tensor of shape (B, C, H, W) containing fractal noise.
    """
    noise = torch.zeros((B, C, H, W))
    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for _ in range(octaves):
        perlin_noise = generate_perlin_noise(B, C, H, W, res_x=(int(res[0] * frequency)), res_y=int(res[1] * frequency))
        noise += perlin_noise * amplitude
        max_amplitude += amplitude
        amplitude *= persistence
        frequency *= 2

    # Normalize the noise
    noise /= max_amplitude

    return noise


def generate_cellular_noise(H, W, num_points=100, *args, **kwargs):
    """
    Generates a 2D Cellular (Worley) Noise pattern using PyTorch.

    Args:
        H, W (int): The height and width of the output noise pattern.
        num_points (int): Number of seed points to generate.

    Returns:
        torch.Tensor: A tensor containing the Cellular noise pattern.
    """
    # Ensure PyTorch is using the same device as the input images
    device = kwargs.get('device', 'cpu')

    # Generate random points (the seeds of the Cellular noise)
    points = torch.rand(num_points, 2, device=device) * torch.tensor([W, H], device=device)

    # Initialize the noise pattern
    noise = torch.full((H, W), float('inf'), device=device)

    # Compute distance from each pixel to the nearest point using broadcasting
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    for point in points:
        dist = torch.sqrt((x_coords - point[0]) ** 2 + (y_coords - point[1]) ** 2)
        noise = torch.min(noise, dist)
    # Normalize the noise pattern
    noise = (noise - torch.min(noise)) / (torch.max(noise) - torch.min(noise))
    return noise


