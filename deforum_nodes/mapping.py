import importlib
import inspect
import sys
"""
DEFORUM STORAGE IMPORT
"""
from .modules.deforum_constants import DeforumStorage

gs = DeforumStorage()

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
from .nodes.deforum_logic_nodes import *
try:
    from .nodes.deforum_noise_nodes import AddCustomNoiseNode
except:
    pass
try:
    from .nodes.deforum_advnoise_node import AddAdvancedNoiseNode
except:
    pass
from .nodes.deforum_prompt_nodes import *
from .nodes.redirect_console_node import DeforumRedirectConsole
from .nodes.deforum_sampler_nodes import *
from .nodes.deforum_schedule_visualizer import *
from .nodes.deforum_video_nodes import *

from . import exec_hijack

# Create an empty dictionary for class mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Iterate through all classes defined in your code
# Get a reference to the current module to inspect its imported node classes
deforum_node_module = sys.modules[__name__]

# Iterate through all classes defined in the current module
for name, obj in inspect.getmembers(deforum_node_module):
    # Check if the member is a class
    if inspect.isclass(obj) and hasattr(obj, "INPUT_TYPES"):
        # Extract the class name and display name
        class_name = name
        display_name = getattr(obj, "display_name", name)  # Use class attribute or default to class name
        # Add the class to the mappings
        NODE_CLASS_MAPPINGS[class_name] = obj
        NODE_DISPLAY_NAME_MAPPINGS[name] = "(deforum) " + display_name
