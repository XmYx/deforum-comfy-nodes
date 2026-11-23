import importlib
import inspect
import os
import sys

"""
DEFORUM STORAGE IMPORT
"""
from .modules.deforum_constants import DeforumStorage

# Built before the node modules are imported: several of them do
# `from ..mapping import gs` at import time.
gs = DeforumStorage()

"""
NODE CLASS IMPORTS

Every module is imported on its own, and a failure only costs that module's
nodes. The audio nodes need librosa and pydub, the advanced noise node needs
opensimplex, the visualizer needs matplotlib — when one of those is missing,
losing a handful of nodes is the honest outcome. Importing them in one flat
list meant a single absent dependency raised out of this module and took all
sixty-odd nodes with it, which is what most of the "IMPORT FAILED" reports on
the tracker actually were.
"""
NODE_MODULES = (
    "deforum_audiosync_nodes",
    "deforum_cache_nodes",
    "deforum_cnet_nodes",
    "deforum_cond_nodes",
    "deforum_data_nodes",
    "deforum_framewarp_node",
    "deforum_guide_nodes",
    "deforum_hybrid_nodes",
    "deforum_interpolation_nodes",
    "deforum_image_nodes",
    "deforum_iteration_nodes",
    "deforum_legacy_nodes",
    "deforum_logic_nodes",
    "deforum_noise_nodes",
    "deforum_advnoise_node",
    "deforum_prompt_nodes",
    "redirect_console_node",
    "deforum_sampler_nodes",
    "deforum_schedule_visualizer",
    "deforum_video_nodes",
)

# name -> exception, for anyone who wants to know what did not make it in.
FAILED_NODE_MODULES = {}
_loaded_modules = []

for _module_name in NODE_MODULES:
    try:
        _loaded_modules.append(
            importlib.import_module(f".nodes.{_module_name}", __package__))
    except Exception as exc:
        FAILED_NODE_MODULES[_module_name] = exc

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _module in _loaded_modules:
    for _name, _obj in inspect.getmembers(_module):
        # __module__ check: node modules import each other's classes, and a
        # class should register once, under the module that defines it.
        if (inspect.isclass(_obj)
                and hasattr(_obj, "INPUT_TYPES")
                and _obj.__module__ == _module.__name__):
            NODE_CLASS_MAPPINGS[_name] = _obj
            NODE_DISPLAY_NAME_MAPPINGS[_name] = (
                "(deforum) " + getattr(_obj, "display_name", _name))

# Without the Iterator there is no animation loop, so nothing else in the pack
# can do anything. Losing it is a core failure, not a missing extra: say so
# instead of registering a handful of nodes that cannot be used for anything.
CORE_NODE_MODULES = ("deforum_iteration_nodes",)

if FAILED_NODE_MODULES:
    _requirements = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "requirements.txt")
    print(f"[deforum] {len(NODE_CLASS_MAPPINGS)} nodes loaded, "
          f"{len(FAILED_NODE_MODULES)} module(s) skipped:")
    for _module_name, _exc in FAILED_NODE_MODULES.items():
        _missing = getattr(_exc, "name", None)
        _reason = (f"needs '{_missing}'" if _missing
                   else f"{type(_exc).__name__}: {_exc}")
        print(f"[deforum]   {_module_name} — {_reason}")
    print(f"[deforum] Install the dependencies into this interpreter with:")
    print(f"[deforum]   {sys.executable} -m pip install -r {_requirements}")

for _module_name in CORE_NODE_MODULES:
    _exc = FAILED_NODE_MODULES.get(_module_name)
    if _exc is not None:
        raise ImportError(
            f"{_module_name} failed to import, so the deforum nodes cannot "
            f"work at all: {_exc}") from _exc
