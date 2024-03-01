import os
import shutil
import sys
import subprocess
import threading
import locale
import traceback
import re


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
        "https://github.com/rgthree/rgthree-comfy",
        "https://github.com/a1lazydog/ComfyUI-AudioScheduler",
        "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet",
        "https://github.com/WASasquatch/was-node-suite-comfyui",
        "https://github.com/11cafe/comfyui-workspace-manager",
        "https://github.com/cubiq/ComfyUI_essentials",
        "https://github.com/FizzleDorf/ComfyUI_FizzNodes",
        "https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation",
        "https://github.com/Fannovel16/ComfyUI-Video-Matting",
        "https://github.com/crystian/ComfyUI-Crystools"
        # Add more repositories as needed
    ]
    comfyui_path = find_path("ComfyUI")
    custom_nodes_path = os.path.join(comfyui_path, 'custom_nodes')



    for repo_url in repositories:
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        repo_dir = os.path.join(custom_nodes_path, repo_name)
        clone_or_pull_repo(repo_url, repo_dir)


try:
    install()

except Exception as e:
    print("[ERROR] deforum-comfy-nodes: Recommended Node package(s) installation failed.")
    traceback.print_exc()
