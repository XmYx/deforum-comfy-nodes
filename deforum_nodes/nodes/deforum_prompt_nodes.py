import torch

from nodes import MAX_RESOLUTION
from ..modules.deforum_node_base import DeforumDataBase

class DeforumPromptNode(DeforumDataBase):
    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"forceInput": False, "multiline": True, "default": "0:'Cat Sushi'"}),
            },
            "optional": {
                "deforum_data": ("deforum_data",),
            },
        }

    RETURN_TYPES = (("deforum_data",))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum"
    display_name = "Prompt"

    @torch.inference_mode()
    def get(self, prompts, deforum_data=None):

        # Splitting the data into rows
        rows = prompts.split('\n')

        # Creating an empty dictionary
        prompts = {}

        # Parsing each row
        for row in rows:
            key, value = row.split(':', 1)
            key = int(key)
            value = value.strip('"')
            prompts[key] = value

        if deforum_data:
            deforum_data["prompts"] = prompts
        else:
            deforum_data = {"prompts": prompts}
        return (deforum_data,)


class DeforumAreaPromptNode(DeforumDataBase):

    default_area_prompt = '[{"0": [{"prompt": "a vast starscape with distant nebulae and galaxies", "x": 0, "y": 0, "w": 1024, "h": 1024, "s": 0.7}, {"prompt": "detailed sci-fi spaceship", "x": 512, "y": 512, "w": 50, "h": 50, "s": 0.7}]}, {"50": [{"prompt": "a vast starscape with distant nebulae and galaxies", "x": 0, "y": 0, "w": 1024, "h": 1024, "s": 0.7}, {"prompt": "detailed sci-fi spaceship", "x": 412, "y": 412, "w": 200, "h": 200, "s": 0.7}]}, {"100": [{"prompt": "a vast starscape with distant nebulae and galaxies", "x": 0, "y": 0, "w": 1024, "h": 1024, "s": 0.7}, {"prompt": "detailed sci-fi spaceship", "x": 112, "y": 112, "w": 800, "h": 800, "s": 0.7}]}]'
    default_prompt = "Alien landscape"

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframe": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "mode":(["default", "percentage", "strength"],),
                "prompt": ("STRING", {"forceInput": False, "multiline": True, 'default': cls.default_prompt,}),
                "width": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 64, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "deforum_data": ("deforum_data",),
            },
        }

    RETURN_TYPES = (("deforum_data",))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum"
    display_name = "Area Prompt"

    @torch.inference_mode()
    def get(self, keyframe, mode, prompt, width, height, x, y, strength, deforum_data=None):

        area_prompt = {"prompt": prompt, "x": x, "y": y, "w": width, "h": height, "s": strength, "mode":mode}
        area_prompt_dict = {f"{keyframe}": [area_prompt]}

        if not deforum_data:
            deforum_data = {"area_prompts":[area_prompt_dict]}

        if "area_prompts" not in deforum_data:
            deforum_data["area_prompts"] = [area_prompt_dict]
        else:

            added = None

            for item in deforum_data["area_prompts"]:
                for k, v in item.items():
                    if int(k) == keyframe:
                        if area_prompt not in v:
                            v.append(area_prompt)
                            added = True
                        else:
                            added = True
            if not added:
                deforum_data["area_prompts"].append(area_prompt_dict)

        deforum_data["prompts"] = None

        return (deforum_data,)
