import os

import cv2
import imageio
import numpy as np
from tqdm import tqdm

import folder_paths
from ..modules.deforum_comfyui_helpers import tensor2pil, pil2tensor, find_next_index, pil_image_to_base64

video_extensions = ['webm', 'mp4', 'mkv', 'gif']

class DeforumLoadVideo:

    def __init__(self):
        self.video_path = None

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in video_extensions):
                    files.append(f)
        return {"required": {
                    "video": (sorted(files),),},}

    CATEGORY = "deforum"
    display_name = "Load Video"

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_video_frame"

    def __init__(self):
        self.cap = None
        self.current_frame = None

    def load_video_frame(self, video):
        video_path = folder_paths.get_annotated_filepath(video)

        # Initialize or reset video capture
        if self.cap is None or self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or self.video_path != video_path:
            try:
                self.cap.release()
            except:
                pass
            self.cap = cv2.VideoCapture(video_path)

            self.cap = cv2.VideoCapture(video_path)
            self.current_frame = -1
            self.video_path = video_path



        success, frame = self.cap.read()
        if success:
            self.current_frame += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame).astype(np.float32)
            frame = pil2tensor(frame)  # Convert to torch tensor
        else:
            # Reset if reached the end of the video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.cap.read()
            self.current_frame = 0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.array(frame).astype(np.float32)
            frame = pil2tensor(frame)  # Convert to torch tensor

        return (frame,)

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        if not folder_paths.exists_annotated_filepath(video):
            return "Invalid video file: {}".format(video)
        return True

class DeforumVideoSaveNode:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.images = []
        self.size = None
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "filename_prefix": ("STRING",{"default":"deforum_"}),
                     "fps": ("INT", {"default": 24, "min": 1, "max": 10000},),
                     "dump_by": (["max_frames", "per_N_frames"],),
                     "dump_every": ("INT", {"default": 0, "min": 0, "max": 4096},),
                     "dump_now": ("BOOLEAN", {"default": False},),
                     "skip_save": ("BOOLEAN", {"default": False},),
                     },
                "optional":
                    {"deforum_frame_data": ("DEFORUM_FRAME_DATA",),}

        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = True

    FUNCTION = "fn"
    display_name = "Save Video"
    CATEGORY = "deforum"
    def add_image(self, image):
        pil_image = tensor2pil(image.unsqueeze(0))
        size = pil_image.size
        if size != self.size:
            self.size = size
            self.images.clear()
        self.images.append(np.array(pil_image).astype(np.uint8))

    def fn(self, image, filename_prefix, fps, dump_by, dump_every, dump_now, skip_save, deforum_frame_data={}):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir)
        counter = find_next_index(full_output_folder, filename_prefix)

        #frame_idx = deforum_frame_data.get["frame_idx"]

        anim_args = deforum_frame_data.get("anim_args")
        if anim_args is not None:
            max_frames = anim_args.max_frames
        else:
            max_frames = image.shape[0] + len(self.images) + 1

        if image.shape[0] > 1:
            for img in image:
                self.add_image(img)
        else:
            self.add_image(image[0])

        print(f"[DEFORUM VIDEO SAVE NODE] holding {len(self.images)} images")
        # When the current frame index reaches the last frame, save the video

        if dump_by == "max_frames":
            dump = len(self.images) >= max_frames
        else:
            dump = len(self.images) >= dump_every

        if dump or dump_now:  # frame_idx is 0-based
            if len(self.images) >= 2:
                if not skip_save:
                    output_path = os.path.join(full_output_folder, f"{filename}_{counter}.mp4")
                    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=10, pixelformat='yuv420p')

                    for frame in tqdm(self.images, desc="Saving MP4 (imageio)"):
                        writer.append_data(frame)
                    writer.close()
                ret = self.images
                self.images = []  # Empty the list for next use
        return {"ui": {"counter":(len(self.images),), "should_dump":(dump or dump_now,), "frames":([pil_image_to_base64(tensor2pil(i)) for i in image]), "fps":(fps,)}, "result": (None if not dump else ret,)}
    @classmethod
    def IS_CHANGED(s, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")