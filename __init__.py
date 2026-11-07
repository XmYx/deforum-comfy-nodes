"""Deforum animation nodes for ComfyUI."""

try:
    from .deforum_nodes.mapping import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError as exc:
    # A bare traceback here is what most of the "IMPORT FAILED" reports on the
    # tracker look like. Say which dependency is missing and how to install it
    # into the interpreter that is actually running ComfyUI.
    import os
    import sys

    missing = getattr(exc, "name", None) or str(exc)
    requirements = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    hint = (
        f'"{missing}" is missing, so the deforum nodes could not load.\n'
        f"Install this pack's dependencies into the Python that runs ComfyUI:\n"
        f"    {sys.executable} -m pip install -r {requirements}\n"
    )
    if missing == "deforum":
        hint += (
            "The deforum backend is a separate package:\n"
            f"    {sys.executable} -m pip install "
            "git+https://github.com/XmYx/deforum-studio.git\n"
        )
    print("[deforum] " + hint.replace("\n", "\n[deforum] "))
    raise ImportError(hint) from exc

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]
