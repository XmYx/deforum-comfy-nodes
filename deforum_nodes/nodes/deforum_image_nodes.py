import cv2
from PIL import Image
import numpy as np
from deforum.generators.deforum_noise_generator import add_noise
from deforum.utils.image_utils import maintain_colors, unsharp_mask, compose_mask_with_check
from ..modules.deforum_comfyui_helpers import tensor2pil, tensor2np, pil2tensor

class DeforumColorMatchNode:

    def __init__(self):
        self.depth_model = None
        self.algo = ""
        self.color_match_sample = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     "force_use_sample": ("BOOLEAN", {"default":False},)
                     },
                "optional":
                    {"force_sample_image":("IMAGE",)}
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fn"
    display_name = "Color Match"
    CATEGORY = "deforum/image"



    def fn(self, image, deforum_frame_data, force_use_sample, force_sample_image=None):
        if image is not None:
            anim_args = deforum_frame_data.get("anim_args")
            image = np.array(tensor2pil(image))
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frame_idx = deforum_frame_data.get("frame_idx", 0)
            if frame_idx == 0 and not force_use_sample:
                self.color_match_sample = None
                return (pil2tensor(image),)
            if force_use_sample:
                if force_sample_image is not None:
                    self.color_match_sample = np.array(tensor2pil(force_sample_image)).copy()
            if anim_args.color_coherence != 'None' and self.color_match_sample is not None:
                image = maintain_colors(image, self.color_match_sample, anim_args.color_coherence)
            print(f"[deforum] ColorMatch: {anim_args.color_coherence}")
            if self.color_match_sample is None:
                self.color_match_sample = image.copy()
            if anim_args.color_force_grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = pil2tensor(image)

        return (image,)


class DeforumAddNoiseNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fn"
    display_name = "Add Noise"
    CATEGORY = "deforum/noise"

    def fn(self, image, deforum_frame_data):

        if image is not None:
            keys = deforum_frame_data.get("keys")
            args = deforum_frame_data.get("args")
            anim_args = deforum_frame_data.get("anim_args")
            root = deforum_frame_data.get("root")
            frame_idx = deforum_frame_data.get("frame_idx")
            noise = keys.noise_schedule_series[frame_idx]
            kernel = int(keys.kernel_schedule_series[frame_idx])
            sigma = keys.sigma_schedule_series[frame_idx]
            amount = keys.amount_schedule_series[frame_idx]
            threshold = keys.threshold_schedule_series[frame_idx]
            contrast = keys.contrast_schedule_series[frame_idx]
            if anim_args.use_noise_mask and keys.noise_mask_schedule_series[frame_idx] is not None:
                noise_mask_seq = keys.noise_mask_schedule_series[frame_idx]
            else:
                noise_mask_seq = None
            mask_vals = {}
            noise_mask_vals = {}

            mask_vals['everywhere'] = Image.new('1', (args.width, args.height), 1)
            noise_mask_vals['everywhere'] = Image.new('1', (args.width, args.height), 1)

            # from ainodes_frontend.nodes.deforum_nodes.deforum_framewarp_node import tensor2np
            prev_img = tensor2np(image)
            mask_image = None
            # apply scaling
            contrast_image = (prev_img * contrast).round().astype(np.uint8)
            # anti-blur
            if amount > 0:
                contrast_image = unsharp_mask(contrast_image, (kernel, kernel), sigma, amount, threshold,
                                              mask_image if args.use_mask else None)
            # apply frame noising
            if args.use_mask or anim_args.use_noise_mask:
                root.noise_mask = compose_mask_with_check(root, args, noise_mask_seq,
                                                          noise_mask_vals,
                                                          Image.fromarray(
                                                              cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)))
            noised_image = add_noise(contrast_image, noise, int(args.seed), anim_args.noise_type,
                                     (anim_args.perlin_w, anim_args.perlin_h,
                                      anim_args.perlin_octaves,
                                      anim_args.perlin_persistence),
                                     root.noise_mask, args.invert_mask)
            # image = Image.fromarray(noised_image)
            print(f"[deforum] Adding Noise {noise} {anim_args.noise_type}")
            image = pil2tensor(noised_image).detach().cpu()

        return (image,)


