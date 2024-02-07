class DeforumDataBase:

    # @classmethod
    # def INPUT_TYPES(s):
    #     return s.params

    RETURN_TYPES = (("deforum_data",))
    FUNCTION = "get"
    OUTPUT_NODE = True
    CATEGORY = f"deforum_data"

    def get(self, deforum_data=None, *args, **kwargs):

        if deforum_data:
            deforum_data.update(**kwargs)
        else:
            deforum_data = kwargs
        print(deforum_data)
        return (deforum_data,)