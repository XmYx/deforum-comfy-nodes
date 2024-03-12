

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
    CATEGORY = f"deforum/cache"
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
    CATEGORY = f"deforum/cache"
    OUTPUT_NODE = True
    display_name = "Load Cached Latent"

    def get_cached_latent(self, cache_index=0):
        from ..mapping import gs
        if gs.reset:
            return (None,)
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
    CATEGORY = f"deforum/cache"
    display_name = "Cache Image"
    OUTPUT_NODE = True

    def cache_it(self, image=None, cache_index=0):
        from ..mapping import gs

        if "image" not in gs.deforum_cache:
            gs.deforum_cache["image"] = {}
        if image is not None:
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
    CATEGORY = f"deforum/cache"
    OUTPUT_NODE = True
    display_name = "Load Cached Image"

    def get_cached_latent(self, cache_index=0):
        from ..mapping import gs

        if gs.reset:
            return (None, None)
        img_dict = gs.deforum_cache.get("image", {})
        image = img_dict.get(cache_index)
        mask = None
        if image is not None:
            mask = image[:, :, :, 0]
        return (image,mask,)




class DeforumCacheStringNode:
    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING",{"default":""}),
                "cache_index": ("INT", {"default":0, "min": 0, "max": 16, "step": 1})
            }
        }

    RETURN_TYPES = (("STRING",))
    FUNCTION = "cache_it"
    CATEGORY = f"deforum/cache"
    display_name = "Cache String"
    OUTPUT_NODE = True

    def cache_it(self, input_string=None, cache_index=0):
        from ..mapping import gs

        if "string" not in gs.deforum_cache:
            gs.deforum_cache["string"] = {}

        gs.deforum_cache["string"][cache_index] = input_string

        return (input_string,)


class DeforumGetCachedStringNode:

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

    RETURN_TYPES = (("STRING",))
    FUNCTION = "get_cached_string"
    CATEGORY = f"deforum/cache"
    OUTPUT_NODE = True
    display_name = "Load Cached String"

    def get_cached_string(self, cache_index=0):
        from ..mapping import gs
        img_dict = gs.deforum_cache.get("string", {})
        string = img_dict.get(cache_index)

        return (str(string),)
