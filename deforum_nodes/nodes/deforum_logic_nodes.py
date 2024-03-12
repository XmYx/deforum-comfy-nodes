

class DeforumImageSwitcherNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "option": ("BOOLEAN", {"default": False}),
            },
            "optional":
                {
                    "image_true": ("IMAGE",),
                    "image_false": ("IMAGE",),

                }
               }

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "compare"
    display_name = "Image Switcher"
    CATEGORY = "deforum/logic"
    OUTPUT_NODE = True

    def compare(self, option=True, image_true=None, image_false=None):

        if option:
            return (image_true,)
        else:
            return (image_false,)


class DeforumComparatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "int_1": ("INT",{"default":0, "min":-999999, "max":2 ** 32, "step":1}),
                "int_2": ("INT",{"default":0, "min":-999999, "max":2 ** 32, "step":1}),
                "condition": (["<", "<=", ">", ">=", "==" ],),
                        }
               }

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "compare"
    display_name = "INT Comparator"
    CATEGORY = "deforum/logic"
    OUTPUT_NODE = True

    def compare(self, int_1, int_2, condition):
        if condition == "<":
            return (int_1 < int_2,)
        elif condition == "<=":
            return (int_1 <= int_2,)
        elif condition == ">":
            return (int_1 > int_2,)
        elif condition == ">=":
            return (int_1 >= int_2,)
        elif condition == "==":
            return (int_1 == int_2,)
        else:
            raise ValueError("Invalid condition")

class DeforumFloatComparatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "float_1": ("FLOAT",{"default":0.00, "min":-999999.00, "max":2 ** 32, "step":0.01}),
                "float_2": ("FLOAT",{"default":0.00, "min":-999999.00, "max":2 ** 32, "step":0.01}),
                "condition": (["<", "<=", ">", ">=", "==" ],),
                        }
               }

    @classmethod
    def IS_CHANGED(cls, text, autorefresh):
        # Force re-evaluation of the node
        if autorefresh == "Yes":
            return float("NaN")

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "compare"
    display_name = "FLOAT Comparator"
    CATEGORY = "deforum/logic"
    OUTPUT_NODE = True

    def compare(self, float_1, float_2, condition):
        if condition == "<":
            return (float_1 < float_2,)
        elif condition == "<=":
            return (float_1 <= float_2,)
        elif condition == ">":
            return (float_1 > float_2,)
        elif condition == ">=":
            return (float_1 >= float_2,)
        elif condition == "==":
            return (float_1 == float_2,)
        else:
            raise ValueError("Invalid condition")


class DeforumAndNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition_1": ("BOOLEAN",),
                "condition_2": ("BOOLEAN",),
                # Add more conditions if needed
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "logical_and"
    display_name = "Logical AND"
    CATEGORY = f"deforum/logic"

    def logical_and(self, condition_1, condition_2, *additional_conditions):
        return (all([condition_1, condition_2] + list(additional_conditions)),)

class DeforumOrNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition_1": ("BOOLEAN",),
                "condition_2": ("BOOLEAN",),
                # Add more conditions if needed
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "logical_or"
    display_name = "Logical OR"
    CATEGORY = f"deforum/logic"

    def logical_or(self, condition_1, condition_2, *additional_conditions):
        return (any([condition_1, condition_2] + list(additional_conditions)),)

class DeforumNotNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN",),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "logical_not"
    display_name = "Logical NOT"
    CATEGORY = f"deforum/logic"

    def logical_not(self, condition):
        return (not condition,)

