class DeforumKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "latent": ("LATENT",),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),

                     "deforum_frame_data": ("DEFORUM_FRAME_DATA",),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    display_name = "KSampler"
    CATEGORY = "deforum"

    def sample(self, model, latent, positive, negative, deforum_frame_data):
        from nodes import common_ksampler
        seed = deforum_frame_data.get("seed", 0)
        steps = deforum_frame_data.get("steps", 10)
        cfg = deforum_frame_data.get("cfg", 7.5)
        sampler_name = deforum_frame_data.get("sampler_name", "euler_a")
        scheduler = deforum_frame_data.get("scheduler", "normal")
        denoise = deforum_frame_data.get("denoise", 1.0)
        latent["samples"] = latent["samples"].float()
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,
                               denoise=denoise)
