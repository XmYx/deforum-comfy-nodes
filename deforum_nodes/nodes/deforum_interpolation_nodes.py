import cv2
import torch
import numpy as np

from deforum import FilmModel
from deforum.models import DepthModel, RAFT
from ..modules.standalone_cadence import CadenceInterpolator
from ..modules.deforum_comfyui_helpers import tensor2pil, pil2tensor

# from ..modules.deforum_constants import deforum_models, deforum_depth_algo
# from .deforum_cache_nodes import deforum_cache

# deforum_models = {}
# deforum_depth_algo = ""

from ..mapping import gs


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
    CATEGORY = "deforum/interpolation"
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
        print(f"[deforum] FILM: {len(return_frames)} frames")
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
                     },
                "optional":{
                    "first_image": ("IMAGE",),
                    "deforum_frame_data": ("DEFORUM_FRAME_DATA",),

                }
                }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGES", "LAST_IMAGE")
    FUNCTION = "fn"
    display_name = "Simple Interpolation"
    CATEGORY = "deforum/interpolation"
    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        # Force re-evaluation of the node
        return float("NaN")

    def interpolate(self, image, method, inter_frames, skip_first, skip_last):

        return_frames = []
        pil_image = tensor2pil(image.clone().detach())
        np_image = np.array(pil_image.convert("RGB"))
        self.FILM_temp.append(np_image)

        print(len(self.FILM_temp))

        if len(self.FILM_temp) == 2:
            if inter_frames > 1:

                if method != "Dyna":

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
                    if not self.model:
                        from ..modules.lvdm.i2v_pipeline import Image2Video
                        self.model = Image2Video()
                    frames = self.model.get_image(self.FILM_temp[0], "cat sushi", steps=50, cfg_scale=7.5, eta=1.0, fs=5, seed=123, image2=self.FILM_temp[1], frames=inter_frames)
                    for frame in frames:
                        tensor = pil2tensor(frame)[0]
                        return_frames.append(tensor)
            else:
                return_frames = [i for i in pil2tensor(self.FILM_temp)[0]]
            self.FILM_temp = [self.FILM_temp[1]]
        print(f"[deforum] Simple Interpolation {len(return_frames)} frames" )
        if len(return_frames) > 0:
            return_frames = torch.stack(return_frames, dim=0)
            return return_frames
        else:
            return None


    def fn(self, image, method, inter_amount, skip_first, skip_last, first_image=None, deforum_frame_data={}):
        last = None
        if deforum_frame_data.get("reset"):
            print("RESETTING DYNA")
            self.FILM_temp = []
        print(image)
        if image is not None:
            result = []
            if image.shape[0] > 1:
                for img in image:
                    interpolated_frames = self.interpolate(img, method, inter_amount, skip_first, skip_last)

                    for f in interpolated_frames:
                        result.append(f)

                ret = torch.stack(result, dim=0)
            else:
                ret = self.interpolate(image[0], method, inter_amount, skip_first, skip_last)
            if ret is not None:
                last = ret[-1].unsqueeze(0)
            return (ret, last,)

        else:
            return (None, None)

class DeforumCadenceNode:
    def __init__(self):
        self.FILM_temp = []
        self.model = None
        self.logger = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "first_image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     "depth_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                     },
                "optional":
                    {"hybrid_images": ("IMAGE",),}

                     }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "fn"
    display_name = "Cadence Interpolation"
    CATEGORY = "deforum/interpolation"

    @classmethod
    def IS_CHANGED(self, *args, **kwargs):
        # Force re-evaluation of the node
        return float("NaN")

    def interpolate(self, image, first_image, deforum_frame_data, depth_strength, dry_run=False, hybrid_images=None):
        self.skip_return = False
        hybrid_provider = None
        #global deforum_depth_algo, deforum_models
        # import turbo_prev_image, turbo_next_image, turbo_next_frame_idx, turbo_prev_frame_idx
        return_frames = []
        if not dry_run:
            pil_image = tensor2pil(image.clone().detach())
            # Convert PIL image to RGB NumPy array and cast to np.float32
            np_image = np.array(pil_image.convert("RGB")).astype(np.uint8)
            # Convert from RGB to BGR for OpenCV compatibility
            #np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            np_image = cv2.normalize(np_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            np_image = np_image.astype(np.uint8)
        else:
            np_image = None
        args = deforum_frame_data["args"]
        anim_args = deforum_frame_data["anim_args"]
        predict_depths = (
                                 anim_args.animation_mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
        predict_depths = predict_depths or (
                anim_args.hybrid_composite and anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth'])

        if "depth_model" not in gs.deforum_models or gs.deforum_depth_algo != anim_args.depth_algorithm:
            self.vram_state = "high"
            if "depth_model" in gs.deforum_models:
                try:
                    gs.deforum_models["depth_model"].to("cpu")
                except:
                    pass
                del gs.deforum_models["depth_model"]

            deforum_depth_algo = anim_args.depth_algorithm
            if predict_depths:
                keep_in_vram = True if self.vram_state == 'high' else False
                # device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else self.root.device)
                # TODO Set device in root in webui
                device = 'cuda'
                gs.deforum_models["depth_model"] = DepthModel("models/other", device,
                                              keep_in_vram=keep_in_vram,
                                              depth_algorithm=anim_args.depth_algorithm, Width=args.width,
                                              Height=args.height,
                                              midas_weight=anim_args.midas_weight)

                # depth-based hybrid composite mask requires saved depth maps
                if anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type == 'Depth':
                    anim_args.save_depth_maps = True
            else:
                gs.deforum_models["depth_model"] = None
                anim_args.save_depth_maps = False
        if gs.deforum_models["depth_model"] != None and not predict_depths:
            gs.deforum_models["depth_model"] = None
        if gs.deforum_models["depth_model"] is not None:
            gs.deforum_models["depth_model"].to('cuda')
        if "raft_model" not in gs.deforum_models:
            gs.deforum_models["raft_model"] = RAFT()
        first_gen = False
        if deforum_frame_data.get("reset") or not hasattr(self, "interpolator"):
            self.interpolator = CadenceInterpolator()
            #deforum_frame_data["frame_idx"] += anim_args.diffusion_cadence
            first_gen = True
        # if self.interpolator.turbo_next_image is not None:
        self.interpolator.turbo_prev_image, self.interpolator.turbo_prev_frame_idx = self.interpolator.turbo_next_image, self.interpolator.turbo_next_frame_idx
        self.interpolator.turbo_next_image, self.interpolator.turbo_next_frame_idx = np_image, deforum_frame_data["frame_idx"]

        if self.interpolator.turbo_next_frame_idx == 0 and first_image is not None:
            pil_image = tensor2pil(first_image.clone().detach())
            # Convert PIL image to RGB NumPy array and cast to np.float32
            np_image = np.array(pil_image.convert("RGB")).astype(np.uint8)
            # Convert from RGB to BGR for OpenCV compatibility
            #np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
            np_image = cv2.normalize(np_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            self.interpolator.turbo_prev_image = np_image.astype(np.uint8)
            self.interpolator.turbo_prev_frame_idx = 0
            self.interpolator.turbo_next_frame_idx = anim_args.diffusion_cadence
            deforum_frame_data["frame_idx"] = anim_args.diffusion_cadence
            self.skip_return = True
        if hybrid_images is not None:
            # try:
            hybrid_provider = []
            for i in hybrid_images:
                pil_image = tensor2pil(i.clone().detach())
                np_image = np.array(pil_image.convert("RGB")).astype(np.uint8)
                np_image = cv2.normalize(np_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                hybrid_provider.append(np_image)
            if len(hybrid_provider) > 0:
                if len(hybrid_provider) - 1 > anim_args.diffusion_cadence:
                    anim_args.diffusion_cadence = len(hybrid_provider) - 1
        # if first_gen:
        #     self.interpolator.turbo_prev_image = np_image
        #     return image
        # with torch.inference_mode():
        if not dry_run:

            with torch.no_grad():
                if not self.logger:
                    import comfy

                    self.logger = comfy.utils.ProgressBar(anim_args.diffusion_cadence)
                # from ..modules.standalone_cadence import new_standalone_cadence
                frames = self.interpolator.new_standalone_cadence(deforum_frame_data["args"],
                                                deforum_frame_data["anim_args"],
                                                deforum_frame_data["root"],
                                                deforum_frame_data["keys"],
                                                deforum_frame_data["frame_idx"],
                                                gs.deforum_models["depth_model"],
                                                gs.deforum_models["raft_model"],
                                                depth_strength,
                                                self.logger,
                                                hybrid_provider=hybrid_provider)



            for frame in frames:
                tensor = pil2tensor(frame)
                return_frames.append(tensor.squeeze(0))
            print(f"[deforum] [rbn] Cadence: {len(return_frames)} frames")
            if len(return_frames) > 0:
                return_frames = torch.stack(return_frames, dim=0)
                return return_frames
            else:
                return None
        else:
            return None

    def fn(self, image, first_image, deforum_frame_data, depth_strength, hybrid_images=None):

        result = []
        ret = None
        if image is not None and not deforum_frame_data.get("reset"):
            # Check if there are multiple images in the batch
            if image.shape[0] > 1:
                for img in image:
                    # Ensure img has batch dimension of 1 for interpolation
                    interpolated_frames = self.interpolate(img.unsqueeze(0), first_image, deforum_frame_data, depth_strength, hybrid_images=hybrid_images)

                    # Collect all interpolated frames
                    for f in interpolated_frames:
                        result.append(f)
                # Stack all results into a single tensor, preserving color channels
                ret = torch.stack(result, dim=0)
            else:
                # Directly interpolate if only one image is present
                ret = self.interpolate(image, first_image, deforum_frame_data, depth_strength, hybrid_images=hybrid_images)
            if ret is not None:
                last = ret[-1].unsqueeze(0)  # Preserve the last frame separately with batch dimension
                if self.skip_return:
                    return (None, last)
                else:
                    return (ret, last,)
        else:
            _ = self.interpolate(None, None, deforum_frame_data, depth_strength, dry_run=True)
            return (None, None,)
