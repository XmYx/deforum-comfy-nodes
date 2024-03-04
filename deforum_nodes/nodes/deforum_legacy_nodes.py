import os
import time

from types import SimpleNamespace
import torch
import numpy as np

from deforum import DeforumAnimationPipeline
from deforum.pipelines.deforum_animation.animation_helpers import DeforumAnimKeys
from deforum.pipelines.deforum_animation.animation_params import RootArgs, DeforumArgs, DeforumAnimArgs, \
    DeforumOutputArgs, LoopArgs, ParseqArgs
from deforum.utils.string_utils import substitute_placeholders

class DeforumSingleSampleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "deforum_data": ("deforum_data",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",)
            },
        }

    RETURN_TYPES = (("IMAGE",))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum"
    display_name = "Integrated Pipeline"

    @torch.inference_mode()
    def get(self, deforum_data, model, clip, vae, *args, **kwargs):

        root_dict = RootArgs()
        args_dict = {key: value["value"] for key, value in DeforumArgs().items()}
        anim_args_dict = {key: value["value"] for key, value in DeforumAnimArgs().items()}
        output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}
        loop_args_dict = {key: value["value"] for key, value in LoopArgs().items()}
        parseq_args_dict = {key: value["value"] for key, value in ParseqArgs().items()}
        root = SimpleNamespace(**root_dict)
        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)
        video_args = SimpleNamespace(**output_args_dict)
        parseq_args = SimpleNamespace(**parseq_args_dict)

        parseq_args.parseq_manifest = ""

        # #parseq_args = None
        loop_args = SimpleNamespace(**loop_args_dict)
        controlnet_args = SimpleNamespace(**{"controlnet_args": "None"})

        for key, value in args.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "":
                    val = None
                else:
                    val = deforum_data[key]
                setattr(args, key, val)

        for key, value in anim_args.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = deforum_data[key]
                setattr(anim_args, key, val)

        for key, value in video_args.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = deforum_data[key]
                setattr(anim_args, key, val)

        for key, value in root.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "":
                    val = None
                else:
                    val = deforum_data[key]
                setattr(root, key, val)

        for key, value in loop_args.__dict__.items():
            if key in deforum_data:
                if deforum_data[key] == "":
                    val = None
                else:
                    val = deforum_data[key]
                setattr(loop_args, key, val)

        success = None
        root.timestring = time.strftime('%Y%m%d%H%M%S')
        args.timestring = root.timestring
        args.strength = max(0.0, min(1.0, args.strength))

        root.animation_prompts = deforum_data.get("prompts", {})


        if not args.use_init and not anim_args.hybrid_use_init_image:
            args.init_image = None

        elif anim_args.animation_mode == 'Video Input':
            args.use_init = True

        current_arg_list = [args, anim_args, video_args, parseq_args, root]
        full_base_folder_path = os.path.join(os.getcwd(), "output/deforum")

        args.batch_name = f"aiNodes_Deforum_{args.timestring}"
        args.outdir = os.path.join(full_base_folder_path, args.batch_name)

        root.raw_batch_name = args.batch_name
        args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)

        # os.makedirs(args.outdir, exist_ok=True)

        def generate(*args, **kwargs):
            from ..modules.deforum_comfy_sampler import sample_deforum
            image = sample_deforum(model, clip, vae, **kwargs)

            return image

        self.deforum = DeforumAnimationPipeline(generate)

        self.deforum.config_dir = os.path.join(os.getcwd(), "output/_deforum_configs")
        os.makedirs(self.deforum.config_dir, exist_ok=True)
        # self.deforum.generate_inpaint = self.generate_inpaint
        import comfy
        pbar = comfy.utils.ProgressBar(deforum_data["max_frames"])

        def datacallback(data=None):
            if data:
                if "image" in data:
                    pbar.update_absolute(data["frame_idx"], deforum_data["max_frames"], ("JPEG", data["image"], 512))

        self.deforum.datacallback = datacallback
        deforum_data["turbo_steps"] = deforum_data.get("diffusion_cadence", 0)
        animation = self.deforum(**deforum_data)

        results = []
        for i in self.deforum.images:
            tensor = torch.from_numpy(np.array(i).astype(np.float32) / 255.0)

            results.append(tensor)
            result = torch.stack(results, dim=0)

        return (result,)


class DeforumSetVAEDownscaleRatioNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"vae": ("VAE",),
                     "downscale_ratio": ("INT", {"default": 42, "min": 32, "max": 64, "step": 1}),
                     },
                }
    RETURN_TYPES = ("VAE",)
    FUNCTION = "fn"
    display_name = "Set VAE Downscale Ratio"
    CATEGORY = "deforum"

    def fn(self, vae, downscale_ratio):
        vae.downscale_ratio = downscale_ratio
        return (vae,)
