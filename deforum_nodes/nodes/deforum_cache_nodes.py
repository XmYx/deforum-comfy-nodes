deforum_cache = {}

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
        global deforum_cache

        if "latent" not in deforum_cache:

            deforum_cache["latent"] = {}

        deforum_cache["latent"][cache_index] = latent

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
        latent_dict = deforum_cache.get("latent", {})
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
        global deforum_cache
        if "image" not in deforum_cache:
            deforum_cache["image"] = {}
        deforum_cache["image"][cache_index] = image.clone()

        print("IMAGE ON INDEX", cache_index, deforum_cache["image"][cache_index])


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

    RETURN_TYPES = (("IMAGE",))
    FUNCTION = "get_cached_latent"
    CATEGORY = f"deforum"
    OUTPUT_NODE = True
    display_name = "Load Cached Image"

    def get_cached_latent(self, cache_index=0):
        global deforum_cache
        img_dict = deforum_cache.get("image", {})
        image = img_dict.get(cache_index)

        print("IMAGE ON INDEX", cache_index, image)

        return (image,)
