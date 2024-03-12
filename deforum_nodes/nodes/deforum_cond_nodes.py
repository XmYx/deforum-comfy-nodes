import random

import torch

from ..modules.deforum_comfyui_helpers import blend_tensors, blend_methods

class DeforumConditioningBlendNode:
    def __init__(self):
        self.prompt = None
        self.n_prompt = None
        self.cond = None
        self.n_cond = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"clip": ("CLIP",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     "blend_method": ([blend_methods]),
                     }
                }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("POSITIVE", "NEGATIVE")
    FUNCTION = "fn"
    display_name = "Blend Conditionings"
    CATEGORY = "deforum/conditioning"
    def fn(self, clip, deforum_frame_data, blend_method):
        areas = deforum_frame_data.get("areas")
        negative_prompt = deforum_frame_data.get("negative_prompt", "")
        n_cond = self.get_conditioning(prompt=negative_prompt, clip=clip)

        if not areas:
            prompt = deforum_frame_data.get("prompt", "")
            next_prompt = deforum_frame_data.get("next_prompt", None)
            cond = self.get_conditioning(prompt=prompt, clip=clip)
            # image = self.getInputData(2)
            # controlnet = self.getInputData(3)

            prompt_blend = deforum_frame_data.get("prompt_blend", 0.0)
            #method = self.content.blend_method.currentText()
            if blend_method != 'none':
                if next_prompt != prompt and prompt_blend != 0.0 and next_prompt is not None:
                    next_cond = self.get_conditioning(prompt=next_prompt, clip=clip)
                    cond = blend_tensors(cond[0], next_cond[0], prompt_blend, blend_method)
                    print(f"[deforum] Blending next prompt: {next_prompt}, with alpha: {prompt_blend} ]")
        else:
            from nodes import ConditioningSetArea
            area_setter = ConditioningSetArea()
            cond = []
            for area in areas:
                prompt = area.get("prompt", None)
                if prompt:

                    new_cond = self.get_conditioning(clip=clip, prompt=area["prompt"])
                    new_cond = area_setter.append(conditioning=new_cond, width=int(area["w"]), height=int(area["h"]), x=int(area["x"]),
                                                  y=int(area["y"]), strength=area["s"])[0]
                    cond += new_cond

        return (cond, n_cond,)

    def get_conditioning(self, prompt="", clip=None, progress_callback=None):


        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]


class DeforumInpaintModelConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),},
                "optional": {
                    "pixels": ("IMAGE",),
                    "mask": ("MASK",),
                    "latent": ("LATENT",),
                    "deforum_frame_data": ("DEFORUM_FRAME_DATA",),

                }

                             }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    display_name = "InpaintModelConditioning [safe]"
    CATEGORY = "deforum/conditioning"
    def encode(self, positive, negative, vae, pixels, mask, latent, deforum_frame_data={}):
        reset = deforum_frame_data.get("reset", False)
        if (pixels is not None and mask is not None) and not reset:
            x = (pixels.shape[1] // 8) * 8
            y = (pixels.shape[2] // 8) * 8
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

            orig_pixels = pixels
            pixels = orig_pixels.clone()
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
                mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

            m = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:,:,:,i] -= 0.5
                pixels[:,:,:,i] *= m
                pixels[:,:,:,i] += 0.5
            concat_latent = vae.encode(pixels)
            orig_latent = vae.encode(orig_pixels)

            out_latent = {}

            out_latent["samples"] = orig_latent
            out_latent["noise_mask"] = mask

            out = []
            for conditioning in [positive, negative]:
                c = []
                for t in conditioning:
                    d = t[1].copy()
                    d["concat_latent_image"] = concat_latent
                    d["concat_mask"] = mask
                    n = [t[0], d]
                    c.append(n)
                out.append(c)
            return (out[0], out[1], out_latent)
        else:
            return (positive, negative, latent,)


class DeforumShuffleTokenizer:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"clip": ("CLIP",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                     }
                }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "fn"
    display_name = "Shuffle Tokenizer"
    CATEGORY = "deforum/conditioning"
    def fn(self, clip, seed=42):
        # Access the tokenizer from the clip object
        tokenizer = clip.tokenizer

        # Copy the original vocabulary for restoration if needed
        original_vocab = tokenizer.vocab.copy()

        # Seed the random number generator for reproducibility
        seeded_random = random.Random(seed)

        # Create a list of (key, value) pairs, shuffle it, then convert it back to a dictionary
        items = list(original_vocab.items())
        seeded_random.shuffle(items)
        shuffled_vocab = dict(items)

        # Update the tokenizer's vocabulary with the shuffled version
        # This step is highly dependent on the tokenizer's implementation.
        # If the tokenizer has a method to set its vocab, use it.
        # Otherwise, you might need to directly set the attribute, if possible.
        # tokenizer.set_vocab(shuffled_vocab)  # Hypothetical method
        tokenizer.vocab = shuffled_vocab  # Direct attribute setting, if no method available

        return (clip,)