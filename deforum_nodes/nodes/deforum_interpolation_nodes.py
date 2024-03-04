import torch
import numpy as np

from deforum import FilmModel
from deforum.models import DepthModel, RAFT
from ..modules.standalone_cadence import CadenceInterpolator
from ..modules.deforum_comfyui_helpers import tensor2pil, pil2tensor

from ..modules.deforum_constants import deforum_models, deforum_depth_algo
from .deforum_cache_nodes import deforum_cache

# deforum_models = {}
# deforum_depth_algo = ""

class DeforumFILMInterpolationNode:
    def __init__(self):
        self.FILM_temp = []
        self.model = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "inter_amount": ("INT", {"default": 2, "min": 1, "max": 10000},),
                     "skip_first": ("BOOLEAN", {"default":True}),
                     "skip_last": ("BOOLEAN", {"default":False}),

                     }
                }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("POSITIVE", "NEGATIVE")
    FUNCTION = "fn"
    display_name = "FILM Interpolation"
    CATEGORY = "deforum"
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        # Force re-evaluation of the node
        return float("NaN")

    def interpolate(self, image, inter_frames, skip_first, skip_last):

        if self.model is None:
            self.model = FilmModel()
            self.model.model.cuda()

        return_frames = []
        pil_image = tensor2pil(image.clone().detach())
        np_image = np.array(pil_image.convert("RGB"))
        self.FILM_temp.append(np_image)
        if len(self.FILM_temp) == 2:

            # with torch.inference_mode():
            with torch.no_grad():
                frames = self.model.inference(self.FILM_temp[0], self.FILM_temp[1], inter_frames=inter_frames)
            # skip_first, skip_last = True, False
            if skip_first:
                frames.pop(0)
            if skip_last:
                frames.pop(-1)

            for frame in frames:
                tensor = pil2tensor(frame)[0]
                return_frames.append(tensor.detach().cpu())
            self.FILM_temp = [self.FILM_temp[1]]
        print(f"[ FILM NODE: Created {len(return_frames)} frames ]")
        if len(return_frames) > 0:
            return_frames = torch.stack(return_frames, dim=0)
            return return_frames
        else:
            return image.unsqueeze(0)


    def fn(self, image, inter_amount, skip_first, skip_last):
        result = []

        if image.shape[0] > 1:
            for img in image:
                interpolated_frames = self.interpolate(img, inter_amount, skip_first, skip_last)

                for f in interpolated_frames:
                    result.append(f)

            ret = torch.stack(result, dim=0)
        else:
            ret = self.interpolate(image[0], inter_amount, skip_first, skip_last)
        return (ret,)

class DeforumSimpleInterpolationNode:
    def __init__(self):
        self.FILM_temp = []
        self.model = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "method": (["DIS Medium", "DIS Fast", "DIS UltraFast", "Farneback Fine", "Normal"],), # "DenseRLOF", "SF",
                     "inter_amount": ("INT", {"default": 2, "min": 1, "max": 10000},),
                     "skip_first": ("BOOLEAN", {"default":False}),
                     "skip_last": ("BOOLEAN", {"default":False}),
                     }
                }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGES", "LAST_IMAGE")
    FUNCTION = "fn"
    display_name = "Simple Interpolation"
    CATEGORY = "deforum"
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        # Force re-evaluation of the node
        return float("NaN")

    def interpolate(self, image, method, inter_frames, skip_first, skip_last):

        return_frames = []
        pil_image = tensor2pil(image.clone().detach())
        np_image = np.array(pil_image.convert("RGB"))
        self.FILM_temp.append(np_image)
        if len(self.FILM_temp) == 2:


            if inter_frames > 1:
                from ..modules.interp import optical_flow_cadence

                frames = optical_flow_cadence(self.FILM_temp[0], self.FILM_temp[1], inter_frames + 1, method)
                # skip_first, skip_last = True, False
                if skip_first:
                    frames.pop(0)
                if skip_last:
                    frames.pop(-1)

                for frame in frames:
                    tensor = pil2tensor(frame)[0]
                    return_frames.append(tensor)
            else:
                return_frames = [i for i in pil2tensor(self.FILM_temp)[0]]
            self.FILM_temp = [self.FILM_temp[1]]
        print(f"[ Simple Interpolation Node: Created {len(return_frames)} frames ]")
        if len(return_frames) > 0:
            return_frames = torch.stack(return_frames, dim=0)
            return return_frames
        else:
            return image.unsqueeze(0)


    def fn(self, image, method, inter_amount, skip_first, skip_last):
        result = []

        if image.shape[0] > 1:
            for img in image:
                interpolated_frames = self.interpolate(img, method, inter_amount, skip_first, skip_last)

                for f in interpolated_frames:
                    result.append(f)

            ret = torch.stack(result, dim=0)
        else:
            ret = self.interpolate(image[0], method, inter_amount, skip_first, skip_last)

        return (ret, ret[-1].unsqueeze(0),)


class DeforumCadenceNode:
    def __init__(self):
        self.FILM_temp = []
        self.model = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     }
                }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "fn"
    display_name = "Cadence Interpolation"
    CATEGORY = "deforum"

    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        # Force re-evaluation of the node
        return float("NaN")

    def interpolate(self, image, deforum_frame_data):
        global deforum_depth_algo, deforum_models
        # import turbo_prev_image, turbo_next_image, turbo_next_frame_idx, turbo_prev_frame_idx
        return_frames = []
        pil_image = tensor2pil(image.clone().detach())
        np_image = np.array(pil_image.convert("RGB"))
        args = deforum_frame_data["args"]
        anim_args = deforum_frame_data["anim_args"]
        predict_depths = (
                                 anim_args.animation_mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
        predict_depths = predict_depths or (
                anim_args.hybrid_composite and anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth'])

        if "depth_model" not in deforum_models or deforum_depth_algo != anim_args.depth_algorithm:
            self.vram_state = "high"
            if "depth_model" in deforum_models:
                deforum_models["depth_model"].to("cpu")
                del deforum_models["depth_model"]

            deforum_depth_algo = anim_args.depth_algorithm
            if predict_depths:
                keep_in_vram = True if self.vram_state == 'high' else False
                # device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else self.root.device)
                # TODO Set device in root in webui
                device = 'cuda'
                deforum_models["depth_model"] = DepthModel("models/other", device,
                                              keep_in_vram=keep_in_vram,
                                              depth_algorithm=anim_args.depth_algorithm, Width=args.width,
                                              Height=args.height,
                                              midas_weight=anim_args.midas_weight)

                # depth-based hybrid composite mask requires saved depth maps
                if anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type == 'Depth':
                    anim_args.save_depth_maps = True
            else:
                deforum_models["depth_model"] = None
                anim_args.save_depth_maps = False
        if deforum_models["depth_model"] != None and not predict_depths:
            deforum_models["depth_model"] = None
        if deforum_models["depth_model"] is not None:
            deforum_models["depth_model"].to('cuda')
        if "raft_model" not in deforum_models:
            deforum_models["raft_model"] = RAFT()

        if deforum_frame_data["frame_idx"] == 0 or not hasattr(self, "interpolator"):
            self.interpolator = CadenceInterpolator()
            deforum_frame_data["frame_idx"] += anim_args.diffusion_cadence
        self.interpolator.turbo_prev_image, self.interpolator.turbo_prev_frame_idx = self.interpolator.turbo_next_image, self.interpolator.turbo_next_frame_idx
        self.interpolator.turbo_next_image, self.interpolator.turbo_next_frame_idx = np_image, deforum_frame_data["frame_idx"]

        # with torch.inference_mode():
        with torch.no_grad():

            # from ..modules.standalone_cadence import new_standalone_cadence
            frames = self.interpolator.new_standalone_cadence(deforum_frame_data["args"],
                                            deforum_frame_data["anim_args"],
                                            deforum_frame_data["root"],
                                            deforum_frame_data["keys"],
                                            deforum_frame_data["frame_idx"],
                                            deforum_models["depth_model"],
                                            deforum_models["raft_model"])



        for frame in frames:

            tensor = pil2tensor(frame)


            return_frames.append(tensor.squeeze(0))
        print(f"[ FILM NODE: Created {len(return_frames)} frames ]")
        if len(return_frames) > 0:
            return_frames = torch.stack(return_frames, dim=0)
            return return_frames
        else:
            return image

    def fn(self, image, deforum_frame_data):
        result = []

        # Check if there are multiple images in the batch
        if image.shape[0] > 1:
            for img in image:
                # Ensure img has batch dimension of 1 for interpolation
                interpolated_frames = self.interpolate(img.unsqueeze(0), deforum_frame_data)

                # Collect all interpolated frames
                for f in interpolated_frames:
                    result.append(f)
            # Stack all results into a single tensor, preserving color channels
            ret = torch.stack(result, dim=0)
        else:
            # Directly interpolate if only one image is present
            ret = self.interpolate(image, deforum_frame_data)

        # The issue might be with how the last frame is being extracted or processed.
        # Ensure the last frame has the correct shape and color channels.
        # Adding unsqueeze(0) to keep the batch dimension consistent.
        last = ret[-1].unsqueeze(0)  # Preserve the last frame separately with batch dimension

        # Ensure the color information is consistent (RGB channels)
        # if last.shape[1] != 3:
        #     # This is just a placeholder check; you might need a different check or fix based on your specific context.
        #     print("Warning: The last frame does not have 3 color channels. Check your interpolate function.")

        return (ret, last,)
