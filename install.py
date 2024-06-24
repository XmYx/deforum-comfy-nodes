import os
import shutil
import sys
import subprocess
import threading
import locale
import traceback
import re
import os
import subprocess
import sys
import platform

if sys.argv[0] == 'install.py':
    sys.path.append('.')   # for portable version


comfy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


sys.path.append(comfy_path)


# ---
def handle_stream(stream, is_stdout):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')

    for msg in stream:
        if is_stdout:
            print(msg, end="", file=sys.stdout)
        else: 
            print(msg, end="", file=sys.stderr)
            

def process_wrap(cmd_str, cwd=None, handler=None):
    print(f"[Deforum] EXECUTE: {cmd_str} in '{cwd}'")
    process = subprocess.Popen(cmd_str, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    if handler is None:
        handler = handle_stream

    stdout_thread = threading.Thread(target=handler, args=(process.stdout, True))
    stderr_thread = threading.Thread(target=handler, args=(process.stderr, False))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()
# ---


pip_list = None


def get_installed_packages():
    global pip_list

    if pip_list is None:
        try:
            result = subprocess.check_output([sys.executable, '-m', 'pip', 'list'], universal_newlines=True)
            pip_list = set([line.split()[0].lower() for line in result.split('\n') if line.strip()])
        except subprocess.CalledProcessError as e:
            print(f"[ComfyUI-Manager] Failed to retrieve the information of installed pip packages.")
            return set()
    
    return pip_list
    

def is_installed(name):
    name = name.strip()
    pattern = r'([^<>!=]+)([<>!=]=?)'
    match = re.search(pattern, name)
    
    if match:
        name = match.group(1)
        
    result = name.lower() in get_installed_packages()
    return result
    

def is_requirements_installed(file_path):
    print(f"req_path: {file_path}")
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if not is_installed(line):
                    return False
                    
    return True
def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def run_git_command(command: str, working_dir: str) -> None:
    """
    Runs a git command in the specified working directory and handles basic errors.
    """
    try:
        subprocess.run(command, shell=True, check=True, cwd=working_dir)
        print(f"Successfully executed: {command} in {working_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {command} in {working_dir}: {e}")


def clone_or_pull_repo(repo_url: str, repo_dir: str) -> None:
    """
    Clones a new repository or updates it if it already exists.
    """
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir, exist_ok=True)
        run_git_command(f"git clone {repo_url} .", repo_dir)
    else:
        run_git_command("git pull", repo_dir)

def install():
    repositories = [
        "https://github.com/ceruleandeep/ComfyUI-LLaVA-Captioner.git",
        "https://github.com/rgthree/rgthree-comfy.git",
        "https://github.com/a1lazydog/ComfyUI-AudioScheduler.git",
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git",
        "https://github.com/WASasquatch/was-node-suite-comfyui.git",
        "https://github.com/11cafe/comfyui-workspace-manager.git",
        "https://github.com/cubiq/ComfyUI_essentials.git",
        "https://github.com/FizzleDorf/ComfyUI_FizzNodes.git",
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
        "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git",
        "https://github.com/Fannovel16/ComfyUI-Video-Matting.git",
        "https://github.com/crystian/ComfyUI-Crystools.git"
        # Add more repositories as needed
    ]
    comfyui_path = find_path("ComfyUI")
    custom_nodes_path = os.path.join(comfyui_path, 'custom_nodes')



    for repo_url in repositories:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_dir = os.path.join(custom_nodes_path, repo_name)
        clone_or_pull_repo(repo_url, repo_dir)


import subprocess


def install_packages():
    # Install packages from requirements.txt
    # subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

    # Force reinstall the deforum-studio package from Git
    subprocess.run(["pip", "install", "--force-reinstall", "git+https://github.com/XmYx/deforum-studio.git"],
                   check=True)


def get_cuda_version():
    try:
        cuda_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        for line in cuda_version.split('\n'):
            if "release" in line:
                return line.split('release')[1].split(',')[0].strip().replace('.', '')
    except Exception as e:
        print(f"Error getting CUDA version: {e}")
        return None

def get_torch_version():
    try:
        import torch
        return torch.__version__.split('+')[0]  # Removes the +cuXXX if exists
    except ImportError:
        print("PyTorch is not installed. Please install PyTorch before proceeding.")
        sys.exit(1)

def construct_wheel_name(cuda_version, py_version, os_name):
    # cuda_version: e.g., "110", "102"
    # py_version: e.g., "38", "37"
    # os_name: either "linux" or "win"
    os_map = {"Linux": "manylinux2014_x86_64", "Windows": "win_amd64"}
    torch_version = get_torch_version().replace('.', '')
    cuda_str = f"cu{cuda_version}" if cuda_version else "cpu"
    filename = f"stable_fast-1.0.4+torch{torch_version}{cuda_str}-cp{py_version}-cp{py_version}-{os_map[os_name]}.whl"
    return filename

def install_stable_fast():
    cuda_version = get_cuda_version()
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    os_name = platform.system()
    wheel_name = construct_wheel_name(cuda_version, python_version, os_name)
    url = f"https://github.com/chengzeyi/stable-fast/releases/download/v1.0.4/{wheel_name}"
    print(f"Attempting to install: {wheel_name}")
    subprocess.run([sys.executable, "-m", "pip", "install", url])



if __name__ == "__main__":
    print("Installing packages...")
    # try:
    #     install_packages()
    #     print("Installation complete.")
    # except Exception as e:
    #     print("[warning] deforum backend package install failed, if you encounter any issues, please activate your venv and run:\npip install git+https://github.com/XmYxdeforum-studio.git\nIf you are using ComfyUI portable, you have to locate your python executable and add that's path before the pip install command.")
    #     pass
    try:
        install_stable_fast()
        print("Installed Stable Fast 1.0.4")
    except:
        print("[warning] Stable Fast Install Failed")
# try:
#     print("")
#     #install()
#
# except Exception as e:
#     print("[ERROR] deforum-comfy-nodes: Recommended Node package(s) installation failed.")
#     traceback.print_exc()