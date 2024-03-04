import importlib
import inspect

"""
NODE CLASS IMPORTS
"""
from .nodes.deforum_audiosync_nodes import *
from .nodes.deforum_cache_nodes import *
from .nodes.deforum_cnet_nodes import *
from .nodes.deforum_cond_nodes import *
from .nodes.deforum_data_nodes import *
from .nodes.deforum_framewarp_node import *
from .nodes.deforum_hybrid_nodes import *
from .nodes.deforum_interpolation_nodes import *
from .nodes.deforum_image_nodes import *
from .nodes.deforum_iteration_nodes import *
from .nodes.deforum_legacy_nodes import *
from .nodes.deforum_prompt_nodes import *
from .nodes.redirect_console_node import DeforumRedirectConsole
from .nodes.deforum_sampler_nodes import *
from .nodes.deforum_video_nodes import *



# Create an empty dictionary for class mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Iterate through all classes defined in your code
# Import the deforum_nodes.deforum_node module
deforum_node_module = importlib.import_module('deforum-comfy-nodes.deforum_nodes.mapping')

# Iterate through all classes defined in deforum_nodes.deforum_node
for name, obj in inspect.getmembers(deforum_node_module):
    # Check if the member is a class
    if inspect.isclass(obj) and hasattr(obj, "INPUT_TYPES"):
        # Extract the class name and display name
        class_name = name
        display_name = getattr(obj, "display_name", name)  # Use class attribute or default to class name
        # Add the class to the mappings
        NODE_CLASS_MAPPINGS[class_name] = obj
        NODE_DISPLAY_NAME_MAPPINGS[name] = "(deforum) " + display_name
