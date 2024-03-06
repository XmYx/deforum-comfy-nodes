import copy

import cv2
import numpy as np
from PIL import Image
from deforum.generators.deforum_flow_generator import get_flow_from_images
from deforum.models import RAFT
from deforum.utils.image_utils import image_transform_optical_flow

from ..modules.deforum_comfyui_helpers import tensor2np, tensor2pil, pil2tensor
from ..modules.deforum_constants import deforum_models, deforum_depth_algo
from .deforum_cache_nodes import deforum_cache


class DeforumApplyFlowNode:
    methods = ['RAFT', 'DIS Medium', 'DIS Fine', 'Farneback']
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "flow_image": ("IMAGE",),
                "flow_method": ([cls.methods]),
                "flow_factor": ("FLOAT", {"default": 0.8, "min": 0, "max": 1.0}),
            },
            "optional":{
                "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
            }
        }

    RETURN_TYPES = (("IMAGE",))
    FUNCTION = "apply_flow"
    CATEGORY = f"deforum"
    OUTPUT_NODE = True
    display_name = "Apply Flow"

    def __init__(self):
        self.image_cache = []

    def apply_flow(self, image, flow_image, flow_method, flow_factor, deforum_frame_data={}):
        global deforum_models
        if "raft_model" not in deforum_models:
            deforum_models["raft_model"] = RAFT()

        if deforum_frame_data.get("reset", None):
            self.image_cache.clear()
        if flow_image is not None:
            temp_np = tensor2np(flow_image)
        else:
            temp_np = tensor2np(image)
        self.image_cache.append(temp_np)

        if len(self.image_cache) >= 2:
            flow = get_flow_from_images(self.image_cache[0], self.image_cache[1], flow_method, deforum_models["raft_model"])
            img = image_transform_optical_flow(tensor2np(image), flow, flow_factor)
            ret = pil2tensor(img)
            self.image_cache = [self.image_cache[1]]
            return (ret,)
        else:
            return (image,)


class DeforumHybridMotionNode:
    raft_model = None
    methods = ['RAFT', 'DIS Medium', 'DIS Fine', 'Farneback']

    def __init__(self):
        self.prev_image = None
        self.flow = None
        self.image_size = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "hybrid_image": ("IMAGE",),
                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     "hybrid_method": ([s.methods]),
                     }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fn"
    display_name = "Hybrid Motion"
    CATEGORY = "deforum"

    def fn(self, image, hybrid_image, deforum_frame_data, hybrid_method):
        if self.raft_model is None:
            self.raft_model = RAFT()

        flow_factor = deforum_frame_data["keys"].hybrid_flow_factor_schedule_series[deforum_frame_data["frame_index"]]
        p_img = tensor2pil(image)
        size = p_img.size

        pil_image = np.array(p_img).astype(np.uint8)

        if self.image_size != size:
            self.prev_image = None
            self.flow = None
            self.image_size = size

        bgr_image = cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)

        if hybrid_image is None:
            if self.prev_image is None:
                self.prev_image = bgr_image
                return (image,)
            else:
                self.flow = get_flow_from_images(self.prev_image, bgr_image, hybrid_method, self.raft_model, self.flow)

                self.prev_image = copy.deepcopy(bgr_image)

                bgr_image = image_transform_optical_flow(bgr_image, self.flow, flow_factor)

                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                return (pil2tensor(rgb_image),)
        else:

            pil_image_ref = np.array(tensor2pil(hybrid_image).resize((self.image_size), Image.Resampling.LANCZOS)).astype(np.uint8)
            bgr_image_ref = cv2.cvtColor(pil_image_ref, cv2.COLOR_RGB2BGR)
            bgr_image_ref = cv2.resize(bgr_image_ref, (bgr_image.shape[1], bgr_image.shape[0]))
            if self.prev_image is None:
                self.prev_image = bgr_image_ref
                return (image,)
            else:
                self.flow = get_flow_from_images(self.prev_image, bgr_image_ref, hybrid_method, self.raft_model,
                                                 self.flow)
                bgr_image = image_transform_optical_flow(bgr_image, self.flow, flow_factor)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                return (pil2tensor(rgb_image),)