class DeforumDataBase:

    # @classmethod
    # def INPUT_TYPES(s):
    #     return s.params

    RETURN_TYPES = (("deforum_data",))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum/parameters"
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Deforum nodes re-run every frame, so always report a new hash.
        return float("NaN")

    def get(self, deforum_data=None, *args, **kwargs):

        if deforum_data:
            deforum_data.update(**kwargs)
        else:
            deforum_data = kwargs
        return (deforum_data,)