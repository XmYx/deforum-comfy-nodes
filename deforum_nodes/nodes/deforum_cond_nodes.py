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
    CATEGORY = "deforum"
    def fn(self, clip, deforum_frame_data, blend_method):
        areas = deforum_frame_data.get("areas")
        negative_prompt = deforum_frame_data.get("negative_prompt", "")
        n_cond = self.get_conditioning(prompt=negative_prompt, clip=clip)

        if not areas:
            prompt = deforum_frame_data.get("prompt", "")
            next_prompt = deforum_frame_data.get("next_prompt", None)
            print(f"[ Deforum Conds: {prompt}, {negative_prompt} ]")
            cond = self.get_conditioning(prompt=prompt, clip=clip)
            # image = self.getInputData(2)
            # controlnet = self.getInputData(3)

            prompt_blend = deforum_frame_data.get("prompt_blend", 0.0)
            #method = self.content.blend_method.currentText()
            if blend_method != 'none':
                if next_prompt != prompt and prompt_blend != 0.0 and next_prompt is not None:
                    next_cond = self.get_conditioning(prompt=next_prompt, clip=clip)
                    cond = blend_tensors(cond[0], next_cond[0], prompt_blend, blend_method)
                    print(f"[ Deforum Cond Blend: {next_prompt}, {prompt_blend} ]")
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
