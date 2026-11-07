import math
import secrets
from types import SimpleNamespace
import torch
from deforum import ImageRNGNoise
from deforum.generators.rng_noise_generator import slerp
from deforum.pipeline_utils import next_seed
from deforum.pipelines.deforum_animation.animation_params import RootArgs, DeforumArgs, DeforumAnimArgs, \
    DeforumOutputArgs, LoopArgs
from deforum.utils.string_utils import split_weighted_subprompts
import comfy.model_management
import comfy.utils

from ..modules.deforum_comfyui_helpers import get_current_keys, generate_seed_list
from ..modules.deforum_compat import LATENT_TYPES, latent_geometry, schedule_value, synchronize

class DeforumIteratorNode:
    def __init__(self):
        self.first_run = True
        self.frame_index = 0
        self.seed = ""
        self.seeds = []
        self.second_run = True
        self.logger = None

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Force re-evaluation of the node
        # if autorefresh == "Yes":
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "deforum_data": ("deforum_data",),
                "latent_type": (LATENT_TYPES, {
                    "default": "auto",
                    "tooltip": "Shape of the starting latent. 'auto' reads it from the "
                               "model input, which is what makes SD3.x, FLUX, "
                               "Qwen-Image and anything newer work unchanged.",
                }),
            },
            "optional": {
                "model": ("MODEL", {
                    "tooltip": "Optional. Connect your model so the starting latent "
                               "matches it automatically.",
                }),
                "latent": ("LATENT",),
                "init_latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "subseed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "subseed_strength": ("FLOAT", {"default": 0.8, "min": 0, "max": 1.0}),
                "slerp_strength": ("FLOAT", {"default": 0.1, "min": 0, "max": 1.0}),
                "prompt_blend_curve": (["none", "linear", "exponential"], {
                    "default": "linear",
                    "tooltip": "Shape of the crossfade between scheduled prompts.",
                }),
                "prompt_blend_frames": ("INT", {
                    "default": 0, "min": 0, "max": 4096,
                    "tooltip": "How many frames before the next prompt keyframe the "
                               "crossfade starts. 0 keeps prompt timing exact.",
                }),
                "reset_counter":("BOOLEAN", {"default": False},),
                "reset_latent":("BOOLEAN", {"default": False},),
                "enable_autoqueue":("BOOLEAN", {"default": False},),
            }
        }

    RETURN_TYPES = (("DEFORUM_FRAME_DATA", "LATENT", "STRING", "STRING"))
    RETURN_NAMES = (("deforum_frame_data", "latent", "positive_prompt", "negative_prompt"))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum/logic"
    display_name = "Iterator Node"


    @torch.inference_mode()
    def get(self, deforum_data, latent_type="auto", model=None, latent=None, init_latent=None, seed=None,
            subseed=None, subseed_strength=None, slerp_strength=None, prompt_blend_curve="linear",
            prompt_blend_frames=0, reset_counter=False, reset_latent=False, enable_autoqueue=False,
            *args, **kwargs):

        from ..mapping import gs
        if gs.reset:
            reset_counter = True
            reset_latent = True

        # global deforum_cache
        root_dict = RootArgs()
        args_dict = {key: value["value"] for key, value in DeforumArgs().items()}
        anim_args_dict = {key: value["value"] for key, value in DeforumAnimArgs().items()}
        output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}
        loop_args_dict = {key: value["value"] for key, value in LoopArgs().items()}
        root = SimpleNamespace(**root_dict)
        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)
        anim_args.diffusion_cadence = 1
        video_args = SimpleNamespace(**output_args_dict)
        parseq_args = None
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

        root.animation_prompts = deforum_data.get("prompts")
        anim_args.fps = 24

        if anim_args.diffusion_cadence and anim_args.diffusion_cadence >= anim_args.max_frames:
            # Stepping by more than the whole animation means the counter is
            # reset on every queue and the run never leaves frame 0.
            print(f"[deforum] diffusion_cadence ({anim_args.diffusion_cadence}) is >= max_frames "
                  f"({anim_args.max_frames}); the animation cannot advance. Raise max_frames or "
                  f"lower the cadence.")

        keys, prompt_series, areas = get_current_keys(anim_args, args.seed, root, area_prompts=deforum_data.get("area_prompts"))
        if self.frame_index >= anim_args.max_frames or reset_counter:
            # if self.logger:
            #     self.logger.stop_live_display()
            # config = {
            #     "Header": {"type": "full", "columns": ["DEFORUM COMFY ANIMATOR - Logging"]},
            #     "Status": {"type": "columns", "columns": ["Frame", "Progress", "Errors"]}
            # }
            # self.logger = TerminalTableLogger(config)
            # self.logger.start_live_display()
            self.reset_counter = False
            self.frame_index = 0
            self.first_run = True
            self.second_run = True

        if not self.logger:
            self.logger = comfy.utils.ProgressBar(anim_args.max_frames)


        self.logger.update_absolute(self.frame_index)

        # else:
        args.scale = schedule_value(keys.cfg_scale_schedule_series, self.frame_index, 7.0)
        if prompt_series is not None:
            args.prompt = prompt_series[self.frame_index]

        seed = schedule_value(keys.seed_schedule_series, self.frame_index, -1)
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = -1

        # -1 is deforum's "pick one for me". It used to be honoured only when
        # seed_behavior was 'random', so the default `0: (-1)` schedule with any
        # other behaviour carried a negative seed into the noise generator,
        # which fills a uint64 array with it: "Python integer -1 out of bounds
        # for uint64".
        if args.seed_behavior == 'random' or seed < 0:
            seed = secrets.randbelow(2 ** 32 - 1)

        self.seed = seed
        args.seed = seed
        self.seeds.append(seed)

        # args.seed = int(args.seed)
        # root.seed_internal = int(root.seed_internal)
        # args.seed_iter_N = int(args.seed_iter_N)
        #
        # if self.seed == "":
        #     self.seed = args.seed
        #     self.seed_internal = root.seed_internal
        #     self.seed_iter_N = args.seed_iter_N
        #
        # self.seed = next_seed(args, root)
        # args.seed = self.seed
        # self.seeds.append(self.seed)

        blend_value = 0.0

        next_frame = self.frame_index + anim_args.diffusion_cadence
        next_prompt = None

        def blend_weight(frames_until_next_change, window, blend_type="linear"):
            """How far to crossfade into the upcoming prompt.

            The crossfade is anchored to the *next* keyframe and only spans
            ``window`` frames, so a prompt scheduled at frame N is rendered
            exactly as written from frame N onwards. With window 0 there is no
            crossfade at all and prompt timing is exact (issue #57).
            """
            if window <= 0 or blend_type == "none":
                return 0.0
            if frames_until_next_change >= window:
                return 0.0
            # 0.0 at the start of the window, 1.0 at the next keyframe
            t = 1.0 - (frames_until_next_change / float(window))
            t = min(max(t, 0.0), 1.0)
            if blend_type == "exponential":
                return 1 / (1 + math.exp(-8 * (t - 0.5)))
            return t

        def find_next_prompt_change(current_index, prompt_series):
            # Step forward from the current position
            for i in range(current_index + 1, len(prompt_series) - 1):
                if i < anim_args.max_frames:

                    if prompt_series[i] != prompt_series[current_index]:
                        return i
            return len(prompt_series) - 1  # default to the end if no change found

        if prompt_series is not None:
            next_prompt_change = find_next_prompt_change(self.frame_index, prompt_series)
            frames_until_change = next_prompt_change - self.frame_index

            blend_value = blend_weight(frames_until_change, prompt_blend_frames, prompt_blend_curve)

            if len(prompt_series) - 1 > next_prompt_change and blend_value > 0.0:
                next_prompt = prompt_series[next_prompt_change]

        gen_args = self.get_current_frame(args, anim_args, root, keys, self.frame_index, areas)

        self.args = args
        self.root = root
        if next_prompt is not None:
            gen_args["next_prompt"], _ = split_weighted_subprompts(str(next_prompt))
            gen_args["prompt_blend"] = blend_value
        gen_args["frame_index"] = self.frame_index
        gen_args["max_frames"] = anim_args.max_frames

        if subseed is None or subseed < 0:
            subseed = seed
        seeds = generate_seed_list(anim_args.max_frames + 1, args.seed_behavior, seed, args.seed_iter_N)
        subseeds = generate_seed_list(anim_args.max_frames + 1, args.seed_behavior, subseed, args.seed_iter_N)
        if reset_counter:
            print("[deforum] RESET COUNTER")
        if reset_latent or not hasattr(self, "rng"):
            print("[deforum] RESET LATENT"  )

            if "image" in gs.deforum_cache:
                gs.deforum_cache["image"].clear()
            if "latent" in gs.deforum_cache:
                gs.deforum_cache["latent"].clear()
            gs.reset = True
            channels, compression = latent_geometry(latent_type, model)
            if init_latent is not None:
                args.height, args.width = (init_latent["samples"].shape[2] * compression,
                                           init_latent["samples"].shape[3] * compression)

            self.rng = ImageRNGNoise((channels, args.height // compression, args.width // compression),
                                     [seeds[self.frame_index]], [subseeds[self.frame_index]],
                                     0.6, 1024, 1024)
            self.latent_compression = compression
            device = comfy.model_management.intermediate_device()
            if channels == 4:
                # SD1.x / SD2.x / SDXL: seed the first frame with real noise so
                # frame 0 is not a grey plate.
                l = self.rng.first().half().to(device)
            else:
                # Everything newer (SD3.x, FLUX, Qwen-Image, Cascade, ...) is
                # happier starting from the empty latent ComfyUI would build, and
                # common_ksampler fills in the right channel count for us.
                l = torch.zeros([1, channels, args.height // compression, args.width // compression],
                                device=device)
            latent = {"samples": l}
            gen_args["denoise"] = 1.0
        else:
            compression = getattr(self, "latent_compression", latent_geometry(latent_type, model)[1])
            if latent is not None and slerp_strength and slerp_strength > 0 and latent["samples"].shape[1] == 4:
                args.height, args.width = (latent["samples"].shape[2] * compression,
                                           latent["samples"].shape[3] * compression)
                l = self.rng.next().clone().to(comfy.model_management.intermediate_device())
                s = latent["samples"].clone().to(comfy.model_management.intermediate_device())
                latent = {"samples":slerp(slerp_strength, s, l)}
        print(f"[deforum] Frame: {self.frame_index} of {anim_args.max_frames}")
        gen_args["noise"] = self.rng
        gen_args["seed"] = int(seed)

        if self.frame_index == 0 and init_latent is not None:
            latent = init_latent
            gen_args["denoise"] = schedule_value(keys.strength_schedule_series, 0, 1.0)

        #if anim_args.diffusion_cadence > 1:
        # global turbo_prev_img, turbo_prev_frame_idx, turbo_next_image, turbo_next_frame_idx, opencv_image
        if anim_args.diffusion_cadence > 1:
            self.frame_index += anim_args.diffusion_cadence if not self.first_run else 0# if anim_args.diffusion_cadence == 1
            if not self.first_run:
                if self.second_run:
                    self.frame_index = 0
                    self.second_run = False
            # if turbo_steps > 1:
            # turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
            # turbo_next_image, turbo_next_frame_idx = opencv_image, self.frame_index
                # frame_idx += turbo_steps

            self.first_run = False

        else:
            self.frame_index += 1 if not self.first_run else 0
            self.first_run = False
            self.second_run = False

        if self.frame_index > anim_args.max_frames:
            self.frame_index = anim_args.max_frames
        if latent is None:
            # The cache can come up empty mid-run (a cleared cache, a reset
            # elsewhere in the graph). Rebuild rather than pass None on.
            print("[deforum] no latent for this frame, rebuilding the starting latent")
            channels, compression = latent_geometry(latent_type, model)
            latent = {"samples": torch.zeros(
                [1, channels, max(1, args.height // compression), max(1, args.width // compression)],
                device=comfy.model_management.intermediate_device())}
            gen_args["denoise"] = 1.0
        latent["samples"] = latent["samples"].float()
        from ..mapping import gs
        gs.reset = False if not self.first_run else True
        enable_autoqueue = enable_autoqueue if self.frame_index == 0 else False
        gen_args["sampler_name"] = deforum_data.get("sampler_name") or "euler_ancestral"
        gen_args["scheduler"] = deforum_data.get("scheduler") or "normal"
        gen_args["reset"] = reset_latent or reset_counter
        gen_args["frame_idx"] = self.frame_index
        gen_args["first_run"] = self.first_run
        gen_args["second_run"] = self.second_run
        gen_args["logger"] = self.logger
        synchronize()

        return {"ui": {"counter":(self.frame_index,), "max_frames":(anim_args.max_frames,), "enable_autoqueue":(enable_autoqueue,)}, "result": (gen_args, latent, gen_args["prompt"], gen_args["negative_prompt"],),}
        # return (gen_args, latent, gen_args["prompt"], gen_args["negative_prompt"],)

    def get_current_frame(self, args, anim_args, root, keys, frame_idx, areas=None):
        if hasattr(args, 'prompt'):
            prompt, negative_prompt = split_weighted_subprompts(args.prompt, frame_idx, anim_args.max_frames)
        else:
            prompt = ""
            negative_prompt = ""
        strength = schedule_value(keys.strength_schedule_series, frame_idx, 1.0)

        return {"prompt": prompt,
                "negative_prompt": negative_prompt,
                "denoise": strength,
                "cfg": args.scale,
                "steps": int(schedule_value(keys.steps_schedule_series, self.frame_index, 20)),
                "root": root,
                "keys": keys,
                "frame_idx": frame_idx,
                "anim_args": anim_args,
                "args": args,
                "areas":areas[frame_idx] if areas is not None else None,
                "logger":self.logger}

class DeforumSeedNode:
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum/logic"
    RETURN_TYPES = (("INT",))
    display_name = "Seed Node"

    @torch.inference_mode()
    def get(self, seed, *args, **kwargs):
        return (seed,)


class DeforumBigBoneResetNode:
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Deforum nodes re-run every frame, so always report a new hash.
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reset_deforum":("BOOLEAN", {"default": False},),},

        }
    FUNCTION = "get"
    CATEGORY = f"deforum/logic"
    RETURN_TYPES = (("BOOLEAN",))
    display_name = "Big Bone Reset Node"

    def get(self, reset_deforum, *args, **kwargs):
        from ..mapping import gs
        gs.reset = reset_deforum
        # deforum_frame_data["reset"] = reset_deforum
        # deforum_frame_data["reset_latent"] = reset_deforum
        # deforum_frame_data["reset_counter"] = reset_deforum
        # deforum_frame_data["first_run"] = reset_deforum
        return {"ui":{"reset":(reset_deforum,)}, "result":(reset_deforum,)}
