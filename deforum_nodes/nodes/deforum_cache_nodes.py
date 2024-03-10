

class DeforumCacheLatentNode:

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "cache_index": ("INT", {"default":0, "min": 0, "max": 16, "step": 1})
            }
        }

    RETURN_TYPES = (("LATENT",))
    FUNCTION = "cache_it"
    CATEGORY = f"deforum"
    display_name = "Cache Latent"
    OUTPUT_NODE = True

    def cache_it(self, latent=None, cache_index=0):
        from ..mapping import gs
        from ..mapping import gs
        if "latent" not in gs.deforum_cache:

            gs.deforum_cache["latent"] = {}

        gs.deforum_cache["latent"][cache_index] = latent

        return (latent,)


class DeforumGetCachedLatentNode:

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {

            "cache_index": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1})

        }}

    RETURN_TYPES = (("LATENT",))
    FUNCTION = "get_cached_latent"
    CATEGORY = f"deforum"
    OUTPUT_NODE = True
    display_name = "Load Cached Latent"

    def get_cached_latent(self, cache_index=0):
        from ..mapping import gs

        latent_dict = gs.deforum_cache.get("latent", {})
        latent = latent_dict.get(cache_index)
        return (latent,)



class DeforumCacheImageNode:
    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "cache_index": ("INT", {"default":0, "min": 0, "max": 16, "step": 1})
            }
        }

    RETURN_TYPES = (("IMAGE",))
    FUNCTION = "cache_it"
    CATEGORY = f"deforum"
    display_name = "Cache Image"
    OUTPUT_NODE = True

    def cache_it(self, image=None, cache_index=0):
        print("CACHING IMAGE")
        from ..mapping import gs

        if "image" not in gs.deforum_cache:
            gs.deforum_cache["image"] = {}

        gs.deforum_cache["image"][cache_index] = image.clone()

        return (image,)


class DeforumGetCachedImageNode:

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {

            "cache_index": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1})

        }}

    RETURN_TYPES = (("IMAGE","MASK"))
    RETURN_NAMES = ("IMAGE","MASK")
    FUNCTION = "get_cached_latent"
    CATEGORY = f"deforum"
    OUTPUT_NODE = True
    display_name = "Load Cached Image"

    def get_cached_latent(self, cache_index=0):
        from ..mapping import gs
        img_dict = gs.deforum_cache.get("image", {})
        image = img_dict.get(cache_index)
        mask = None
        if image is not None:
            mask = image[:, :, :, 0]
        return (image,mask,)
