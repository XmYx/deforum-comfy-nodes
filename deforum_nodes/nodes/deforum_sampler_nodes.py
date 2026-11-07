from ..modules.deforum_compat import block, resolve_sampler_name, resolve_scheduler_name


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
    CATEGORY = "deforum/sampling"

    def sample(self, model, latent, positive, negative, deforum_frame_data):
        from nodes import common_ksampler
        if latent is None:
            # Usually an empty deforum cache on the first frame of a run. Stop
            # this branch quietly rather than raising deep inside the sampler.
            print("[deforum] KSampler got no latent, skipping this frame")
            return (block(),)
        seed = int(deforum_frame_data.get("seed") or 0)
        steps = max(1, int(deforum_frame_data.get("steps") or 20))
        cfg = float(deforum_frame_data.get("cfg") or 7.5)
        sampler_name = resolve_sampler_name(deforum_frame_data.get("sampler_name"))
        scheduler = resolve_scheduler_name(deforum_frame_data.get("scheduler"))
        denoise = deforum_frame_data.get("denoise")
        denoise = 1.0 if denoise is None else float(denoise)
        # A fully denoised pass with a strength schedule of 0 would just redraw
        # noise; clamp so a schedule that dips to 0 does not blank the frame.
        denoise = min(max(denoise, 0.01), 1.0)

        latent = dict(latent)
        latent["samples"] = latent["samples"].float()
        return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,
                               denoise=denoise)


class DeforumVAEEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                              "vae": ("VAE", )},
                "optional": {
                            "pixels": ("IMAGE",),
                            "latent": ("LATENT",)
                }


                }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    display_name = "VAEEncode [safe]"

    CATEGORY = "deforum/latent"

    def encode(self, vae, pixels=None, latent=None):
        if pixels is not None:
            t = vae.encode(pixels[:, :, :, :3])
            return ({"samples": t}, )
        if latent is not None:
            return (latent,)
        print("[deforum] VAEEncode got neither pixels nor a latent, skipping this frame")
        return (block(),)
