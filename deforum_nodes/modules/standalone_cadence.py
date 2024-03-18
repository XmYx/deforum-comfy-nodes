import os

import cv2
import numpy as np
import torch

from deforum.generators.deforum_flow_generator import rel_flow_to_abs_flow, abs_flow_to_rel_flow, get_flow_from_images, \
    get_flow_for_hybrid_motion_prev_imgs
from deforum.utils.deforum_framewarp_utils import anim_frame_warp
from deforum.utils.deforum_hybrid_animation import get_matrix_for_hybrid_motion_prev_imgs
from deforum.utils.image_utils import image_transform_optical_flow, image_transform_ransac
import cv2
import numpy as np
import PIL.Image

class CadenceInterpolator:
    def __init__(self):
        self.turbo_prev_image, self.turbo_prev_frame_idx = None, 0
        self.turbo_next_image, self.turbo_next_frame_idx = None, 0
        self.start_frame = 0
        self.depth = None
        self.prev_img = None
        self.prev_flow = None

    def new_standalone_cadence(self, args, anim_args, root, keys, frame_idx, depth_model, raft_model, depth_strength=1.0, logger=None, hybrid_provider=None):
        # global self.turbo_prev_image, self.turbo_next_image, self.turbo_prev_frame_idx, self.turbo_next_frame_idx, self.start_frame, self.depth
        if not hasattr(self, 'prev_flow'):
            self.prev_flow = None
        self.prev_img = self.turbo_prev_image
        goal_img = self.turbo_next_image
        turbo_steps = int(anim_args.diffusion_cadence)
        cadence_flow_factor = keys.cadence_flow_factor_schedule_series[frame_idx]
        hybrid_flow_factor = keys.hybrid_flow_factor_schedule_series[frame_idx]
        hybrid_comp_schedules = keys.hybrid_comp_alpha_schedule_series[frame_idx]
        imgs = []
        x = 0
        # emit in-between frames
        if turbo_steps > 1:
            tween_frame_start_idx = max(self.start_frame, frame_idx - turbo_steps)

            cadence_flow = None
            for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                # update progress during cadence
                # state.job = f"frame {tween_frame_idx + 1}/{anim_args.max_frames}"
                # state.job_no = tween_frame_idx + 1
                # cadence vars
                tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
                advance_prev = self.turbo_prev_image is not None and tween_frame_idx > self.turbo_prev_frame_idx
                advance_next = tween_frame_idx > self.turbo_next_frame_idx
    
                # optical flow cadence setup before animation warping
                if anim_args.animation_mode in ['2D', '3D'] and anim_args.optical_flow_cadence != 'None':
                    if keys.strength_schedule_series[tween_frame_start_idx] > 0:
                        if cadence_flow is None and self.turbo_prev_image is not None and self.turbo_next_image is not None:

                            cadence_flow = get_flow_from_images(self.turbo_prev_image, self.turbo_next_image,
                                                                anim_args.optical_flow_cadence, raft_model) / 2
                            self.turbo_next_image = image_transform_optical_flow(self.turbo_next_image, -cadence_flow, 1)
    
                # if opts.data.get("deforum_save_gen_info_as_srt"):
                #     params_to_print = opts.data.get("deforum_save_gen_info_as_srt_params", ['Seed'])
                #     params_string = format_animation_params(keys, prompt_series, tween_frame_idx, params_to_print)
                #     write_frame_subtitle(srt_filename, tween_frame_idx, srt_frame_duration,
                #                          f"F#: {tween_frame_idx}; Cadence: {tween < 1.0}; Seed: {args.seed}; {params_string}")
                #     params_string = None
                # if logger:
                #     logger.update_row("Status", ["Cadence", str(tween_frame_idx), f"tween:{tween:0.2f}"])

                    # print(
                    #     f"[deforum] Creating in-between {'' if cadence_flow is None else anim_args.optical_flow_cadence + ' optical flow '}cadence frame: {tween_frame_idx}; tween:{tween:0.2f};")
    
                if depth_model is not None:
                    assert (self.turbo_next_image is not None)
                    self.depth = depth_model.predict(self.turbo_next_image, anim_args.midas_weight, root.half_precision) * depth_strength
    
                if advance_prev:
                    self.turbo_prev_image, _, _ = anim_frame_warp(self.turbo_prev_image, args, anim_args, keys, tween_frame_idx,
                                                          depth_model, depth=self.depth, device=root.device,
                                                          half_precision=root.half_precision)
                if advance_next:
                    self.turbo_next_image, _, _ = anim_frame_warp(self.turbo_next_image, args, anim_args, keys, tween_frame_idx,
                                                          depth_model, depth=self.depth, device=root.device,
                                                          half_precision=root.half_precision)
    
                # hybrid video motion - warps self.turbo_prev_image or self.turbo_next_image to match motion
                if tween_frame_idx > 0:
                    if anim_args.hybrid_motion in ['Affine', 'Perspective']:
                        if anim_args.hybrid_motion_use_prev_img:
                            matrix = get_matrix_for_hybrid_motion_prev_imgs(tween_frame_idx - 1, (args.width, args.height),
                                                                       self.turbo_next_image, self.turbo_prev_image, anim_args.hybrid_motion)
                            if advance_prev:
                                self.turbo_prev_image = image_transform_ransac(self.turbo_prev_image, matrix, anim_args.hybrid_motion)
                            if advance_next:
                                self.turbo_next_image = image_transform_ransac(self.turbo_next_image, matrix, anim_args.hybrid_motion)
                        else:
                            if hybrid_provider is not None:
                                # matrix = get_matrix_for_hybrid_motion(tween_frame_idx - 1, (args.W, args.H), inputfiles,
                                #                                       anim_args.hybrid_motion)
                                matrix = get_matrix_for_hybrid_motion_prev_imgs(tween_frame_idx - 1,
                                                                                (args.width, args.height),
                                                                                hybrid_provider[x + 1], hybrid_provider[x + 1],
                                                                                anim_args.hybrid_motion)
                                if advance_prev:
                                    self.turbo_prev_image = image_transform_ransac(self.turbo_prev_image, matrix, anim_args.hybrid_motion)
                                if advance_next:
                                    self.turbo_next_image = image_transform_ransac(self.turbo_next_image, matrix, anim_args.hybrid_motion)
                    if anim_args.hybrid_motion in ['Optical Flow']:
                        if anim_args.hybrid_motion_use_prev_img:

                            flow = get_flow_for_hybrid_motion_prev_imgs(self.turbo_next_image,
                                        self.prev_flow,
                                        self.turbo_prev_image,
                                        anim_args.hybrid_flow_method,
                                        raft_model)


                            if advance_prev:
                                self.turbo_prev_image = image_transform_optical_flow(self.turbo_prev_image, flow,
                                                                                #hybrid_comp_schedules['flow_factor'])
                                                                                hybrid_flow_factor)
                            if advance_next:
                                self.turbo_next_image = image_transform_optical_flow(self.turbo_next_image, flow,
                                                                                #hybrid_comp_schedules['flow_factor'])
                                                                                hybrid_flow_factor)
                            self.prev_flow = flow
                        else:
                            if hybrid_provider is not None:
                                flow = get_flow_for_hybrid_motion_prev_imgs(hybrid_provider[x + 1],
                                                                            self.prev_flow,
                                                                            hybrid_provider[x],
                                                                            anim_args.hybrid_flow_method,
                                                                            raft_model)

                                if advance_prev:
                                    self.turbo_prev_image = image_transform_optical_flow(self.turbo_prev_image, flow,
                                                                                    hybrid_flow_factor)
                                if advance_next:
                                    self.turbo_next_image = image_transform_optical_flow(self.turbo_next_image, flow,
                                                                                    hybrid_flow_factor)
                                self.prev_flow = flow
    
                # do optical flow cadence after animation warping
                if cadence_flow is not None:
                    cadence_flow = abs_flow_to_rel_flow(cadence_flow, args.width, args.height)
                    cadence_flow, _, _ = anim_frame_warp(cadence_flow, args, anim_args, keys, tween_frame_idx, depth_model,
                                                      depth=self.depth, device=root.device, half_precision=root.half_precision)
                    cadence_flow_inc = rel_flow_to_abs_flow(cadence_flow, args.width, args.height) * tween
                    if advance_prev:
                        self.turbo_prev_image = image_transform_optical_flow(self.turbo_prev_image, cadence_flow_inc,
                                                                        cadence_flow_factor)
                    if advance_next:
                        self.turbo_next_image = image_transform_optical_flow(self.turbo_next_image, cadence_flow_inc,
                                                                        cadence_flow_factor)
    
                self.turbo_prev_frame_idx = self.turbo_next_frame_idx = tween_frame_idx
    
                if self.turbo_prev_image is not None and tween < 1.0:
                    img = self.turbo_prev_image * (1.0 - tween) + self.turbo_next_image * tween
                else:
                    img = self.turbo_next_image
    
                # intercept and override to grayscale
                if anim_args.color_force_grayscale:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
                    # overlay mask
                # if args.overlay_mask and (anim_args.use_mask_video or args.use_mask):
                #     img = do_overlay_mask(args, anim_args, img, tween_frame_idx, True)
                if logger:
                    preview = preview
                    pil_img = PIL.Image.fromarray(img.astype(np.uint8))
                    logger.update_absolute(turbo_steps - (frame_idx - tween_frame_idx) + 1, preview=("JPEG", pil_img, 512))

                imgs.append(img)
                x += 1
        return imgs
    
    
    




#
# def standalone_cadence(img, prev_img, frame_idx, cadence_frames, args, anim_args, keys, raft_model=None, depth_model=None):
#     if img is not None:
#         if cadence_frames > 1:
#             if isinstance(img, PIL.Image.Image):
#                 img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#             if isinstance(prev_img, PIL.Image.Image):
#                 prev_img = cv2.cvtColor(np.array(prev_img), cv2.COLOR_RGB2BGR)
#
#             tween_frame_start_idx = max(0, frame_idx - cadence_frames)
#             cadence_flow = None
#             return_frames = []
#             for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
#                 tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
#                 advance_prev = prev_img is not None and tween_frame_idx > 0
#                 advance_next = tween_frame_idx > 0
#
#                 if anim_args.animation_mode in ['2D', '3D'] and anim_args.optical_flow_cadence != 'None':
#                     if keys.strength_schedule_series[tween_frame_start_idx] > 0:
#                         if cadence_flow is None and prev_img is not None and img is not None:
#                             cadence_flow = get_flow_from_images(prev_img, img, anim_args.optical_flow_cadence, raft_model) / 2
#                             img = image_transform_optical_flow(img, -cadence_flow, 1)
#
#                 if depth_model is not None:
#                     depth = depth_model.predict(img, anim_args.midas_weight, True)
#                     if advance_prev:
#                         prev_img, _, _ = anim_frame_warp(prev_img, args, anim_args, keys, tween_frame_idx, depth_model=depth_model, depth=self.depth, device='cuda', half_precision=True)
#                     if advance_next:
#                         img, _, _ = anim_frame_warp(img, args, anim_args, keys, tween_frame_idx, depth_model=depth_model, depth=self.depth, device='cuda', half_precision=True)
#
#                 if cadence_flow is not None:
#                     cadence_flow = abs_flow_to_rel_flow(cadence_flow, args.width, args.height)
#                     cadence_flow, _, _ = anim_frame_warp(cadence_flow, args, anim_args, keys, tween_frame_idx, depth_model=depth_model, depth=self.depth, device='cuda', half_precision=True)
#                     cadence_flow_inc = rel_flow_to_abs_flow(cadence_flow, args.width, args.height) * tween
#                     if advance_prev:
#                         prev_img = image_transform_optical_flow(prev_img, cadence_flow_inc, 1)
#                     if advance_next:
#                         img = image_transform_optical_flow(img, cadence_flow_inc, 1)
#
#                 if prev_img is not None and tween < 1.0:
#                     combined_img = prev_img * (1.0 - tween) + img * tween
#                 else:
#                     combined_img = img
#
#                 if anim_args.color_force_grayscale:
#                     combined_img = cv2.cvtColor(combined_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
#                     combined_img = cv2.cvtColor(combined_img, cv2.COLOR_GRAY2BGR)
#                 res = combined_img.astype(np.uint8)
#                 return_frames.append(res)
#             return return_frames, prev_img, img

