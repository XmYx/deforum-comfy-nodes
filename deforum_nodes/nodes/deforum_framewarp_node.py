
import cv2
import numpy as np
import torch
from PIL import Image
from einops import rearrange, repeat

import comfy
from ..modules.deforum_comfyui_helpers import tensor2np, pil2tensor


class DeforumFrameWarpNode:
    def __init__(self):
        self.depth_model = None
        self.depth = None
        self.algo = ""
        self.depth_min, self.depth_max = 1000, -1000
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     "warp_depth_image": ("BOOLEAN",{"default":False}),
                     },
                "optional":
                    {
                        "depth_image":("IMAGE",),
                    }
                }

    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE")
    RETURN_NAMES = ("IMAGE","DEPTH", "WARPED_DEPTH")
    FUNCTION = "fn"
    display_name = "Frame Warp"
    CATEGORY = "deforum/image"

    def fn(self, image, deforum_frame_data, warp_depth_image, depth_image=None):
        from deforum.models import DepthModel
        from deforum.utils.deforum_framewarp_utils import anim_frame_warp
        np_image = None
        data = deforum_frame_data
        if image is not None:
            if image.shape[0] > 1:
                for img in image:
                    np_image = tensor2np(img)
            else:
                np_image = tensor2np(image)

            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

            args = data.get("args")
            anim_args = data.get("anim_args")
            keys = data.get("keys")
            frame_idx = data.get("frame_idx")

            if frame_idx == 0:
                self.depth = None
            predict_depths = (
                                     anim_args.animation_mode == '3D' and anim_args.use_depth_warping) or anim_args.save_depth_maps
            predict_depths = predict_depths or (
                    anim_args.hybrid_composite and anim_args.hybrid_comp_mask_type in ['Depth', 'Video Depth'])


            if depth_image is not None:
                predict_depths = False
            if self.depth_model == None or self.algo != anim_args.depth_algorithm:
                self.vram_state = "high"
                if self.depth_model is not None:
                    self.depth_model.to("cpu")
                    del self.depth_model
                    # torch_gc()

                self.algo = anim_args.depth_algorithm
                if predict_depths:
                    keep_in_vram = True if self.vram_state == 'high' else False
                    # device = ('cpu' if cmd_opts.lowvram or cmd_opts.medvram else self.root.device)
                    # TODO Set device in root in webui
                    device = 'cuda'
                    self.depth_model = DepthModel("models/other", device,
                                                  keep_in_vram=keep_in_vram,
                                                  depth_algorithm=anim_args.depth_algorithm, Width=args.width,
                                                  Height=args.height,
                                                  midas_weight=anim_args.midas_weight)

                    # depth-based hybrid composite mask requires saved depth maps
                    if anim_args.hybrid_composite != 'None' and anim_args.hybrid_comp_mask_type == 'Depth':
                        anim_args.save_depth_maps = True
                else:
                    self.depth_model = None
                    anim_args.save_depth_maps = False
            if self.depth_model != None and not predict_depths:
                self.depth_model = None
            if self.depth_model is not None:
                self.depth_model.to('cuda')
            if depth_image is not None:

                depth_image = comfy.utils.common_upscale(depth_image.permute(0,3,1,2), args.width, args.height, upscale_method="bislerp", crop="disabled")
                depth_image = depth_image.permute(0,2,3,1) * 255.0

                if depth_image.dim() > 2:

                    depth_image = depth_image[0].mean(dim=-1)  # Take the mean across the color channels

            warped_np_img, depth, mask = anim_frame_warp(np_image, args, anim_args, keys, frame_idx,
                                                              depth_model=self.depth_model, depth=depth_image, device='cuda',
                                                              half_precision=True)
            image = Image.fromarray(cv2.cvtColor(warped_np_img, cv2.COLOR_BGR2RGB))
            tensor = pil2tensor(image)
            if depth is not None:
                num_channels = len(depth.shape)

                if num_channels <= 3:
                    depth_image_pil = self.to_image(depth.detach().cpu())
                else:
                    depth_image_pil = self.to_image(depth[0].detach().cpu())


                ret_depth = pil2tensor(depth_image_pil).detach().cpu()
                if warp_depth_image:
                    warped_depth, _, _ = anim_frame_warp(np.array(depth_image_pil), args, anim_args, keys, frame_idx,
                                                                      depth_model=self.depth_model, depth=depth_image,
                                                                      device='cuda',
                                                                      half_precision=True)
                    warped_depth_image = Image.fromarray(warped_depth)
                    warped_ret = pil2tensor(warped_depth_image).detach().cpu()
                else:
                    warped_ret = ret_depth

            else:
                ret_depth = tensor
                warped_ret = tensor
            self.depth = depth

            # if gs.vram_state in ["low", "medium"] and self.depth_model is not None:
            #     self.depth_model.to('cpu')


            # if mask is not None:
            #     mask = mask.detach().cpu()
            #     # mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            #     mask = mask.mean(dim=0, keepdim=False)
            #     mask[mask > 1e-05] = 1
            #     mask[mask < 1e-05] = 0
            #     mask = mask[0].unsqueeze(0)



            # from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.resizeRight import resizeright
            # from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.resizeRight import interp_methods
            # mask = resizeright.resize(mask, scale_factors=None,
            #                                     out_shape=[mask.shape[0], int(mask.shape[1] // 8), int(mask.shape[2] // 8)
            #                                             ],
            #                                     interp_method=interp_methods.lanczos3, support_sz=None,
            #                                     antialiasing=True, by_convs=True, scale_tolerance=None,
            #                                     max_numerator=10, pad_mode='reflect')
            return (tensor, ret_depth,warped_ret,)
            # return [data, tensor, mask, ret_depth, self.depth_model]
        else:
            return (image, image,image,)
    def to_image(self, depth: torch.Tensor):
        depth = depth.cpu().numpy()
        depth = np.expand_dims(depth, axis=0) if len(depth.shape) == 2 else depth
        self.depth_min, self.depth_max = min(self.depth_min, depth.min()), max(self.depth_max, depth.max())
        denom = max(1e-8, self.depth_max - self.depth_min)
        temp = rearrange((depth - self.depth_min) / denom * 255, 'c h w -> h w c')
        return Image.fromarray(repeat(temp, 'h w 1 -> h w c', c=3).astype(np.uint8))