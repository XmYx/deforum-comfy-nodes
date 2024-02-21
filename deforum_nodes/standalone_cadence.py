import cv2
import numpy as np
from deforum.generators.deforum_flow_generator import rel_flow_to_abs_flow, abs_flow_to_rel_flow, get_flow_from_images
from deforum.utils.deforum_framewarp_utils import anim_frame_warp
from deforum.utils.image_utils import image_transform_optical_flow



import cv2
import numpy as np
import PIL.Image

def standalone_cadence(img, prev_img, frame_idx, cadence_frames, args, anim_args, keys, raft_model=None, depth_model=None):
    if img is not None:
        if cadence_frames > 1:
            if isinstance(img, PIL.Image.Image):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            if isinstance(prev_img, PIL.Image.Image):
                prev_img = cv2.cvtColor(np.array(prev_img), cv2.COLOR_RGB2BGR)

            tween_frame_start_idx = max(0, frame_idx - cadence_frames)
            cadence_flow = None
            return_frames = []
            for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
                advance_prev = prev_img is not None and tween_frame_idx > 0
                advance_next = tween_frame_idx > 0

                if anim_args.animation_mode in ['2D', '3D'] and anim_args.optical_flow_cadence != 'None':
                    if keys.strength_schedule_series[tween_frame_start_idx] > 0:
                        if cadence_flow is None and prev_img is not None and img is not None:
                            cadence_flow = get_flow_from_images(prev_img, img, anim_args.optical_flow_cadence, raft_model) / 2
                            img = image_transform_optical_flow(img, -cadence_flow, 1)

                if depth_model is not None:
                    depth = depth_model.predict(img, anim_args.midas_weight, True)
                    if advance_prev:
                        prev_img, _, _ = anim_frame_warp(prev_img, args, anim_args, keys, tween_frame_idx, depth_model=depth_model, depth=depth, device='cuda', half_precision=True)
                    if advance_next:
                        img, _, _ = anim_frame_warp(img, args, anim_args, keys, tween_frame_idx, depth_model=depth_model, depth=depth, device='cuda', half_precision=True)

                if cadence_flow is not None:
                    cadence_flow = abs_flow_to_rel_flow(cadence_flow, args.width, args.height)
                    cadence_flow, _, _ = anim_frame_warp(cadence_flow, args, anim_args, keys, tween_frame_idx, depth_model=depth_model, depth=depth, device='cuda', half_precision=True)
                    cadence_flow_inc = rel_flow_to_abs_flow(cadence_flow, args.width, args.height) * tween
                    if advance_prev:
                        prev_img = image_transform_optical_flow(prev_img, cadence_flow_inc, 1)
                    if advance_next:
                        img = image_transform_optical_flow(img, cadence_flow_inc, 1)

                if prev_img is not None and tween < 1.0:
                    combined_img = prev_img * (1.0 - tween) + img * tween
                else:
                    combined_img = img

                if anim_args.color_force_grayscale:
                    combined_img = cv2.cvtColor(combined_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    combined_img = cv2.cvtColor(combined_img, cv2.COLOR_GRAY2BGR)
                res = combined_img.astype(np.uint8)
                return_frames.append(res)
            return return_frames, prev_img, img

