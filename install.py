"""Optional post-install step for the deforum ComfyUI nodes.

Everything this pack actually needs lives in ``requirements.txt``, which
ComfyUI-Manager and ``comfy node install`` install for you. This script exists
only to install the deforum backend into the *same* interpreter that runs
ComfyUI, for people who cloned the repo by hand.

Deliberately NOT done here any more:

* cloning a dozen unrelated node packs into ``custom_nodes`` - that is the
  user's choice, and it used to silently rewrite their install;
* installing ``stable-fast`` wheels - the prebuilt wheels are CUDA-only and
  pulled a mismatched torch onto AMD and Apple machines (issue #55);
* pinning ``numpy<2`` / ``moviepy<2`` - that downgraded working ComfyUI
  installs (issues #51, #62, #70).
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "requirements.txt")

DEFORUM_BACKEND = "git+https://github.com/XmYx/deforum-studio.git"


def pip(*args):
    cmd = [sys.executable, "-m", "pip", *args]
    print(f"[deforum] {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    print(f"[deforum] installing into {sys.executable}")

    if os.path.exists(REQUIREMENTS):
        if pip("install", "-r", REQUIREMENTS) == 0:
            print("[deforum] requirements installed")
            return 0
        print("[deforum] installing requirements.txt failed, trying the backend on its own")

    if pip("install", "--upgrade", DEFORUM_BACKEND) != 0:
        print(
            "[deforum] Could not install the deforum backend automatically.\n"
            "[deforum] Activate the environment that runs ComfyUI and run:\n"
            f"[deforum]     pip install -r {REQUIREMENTS}\n"
            "[deforum] On ComfyUI portable, use the bundled interpreter, e.g.\n"
            "[deforum]     python_embeded\\python.exe -m pip install -r "
            "ComfyUI\\custom_nodes\\deforum-comfy-nodes\\requirements.txt"
        )
        return 1

    print("[deforum] backend installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
