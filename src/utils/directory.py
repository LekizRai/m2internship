from pathlib import Path


def get_project_root_folder() -> str:
    # Get current directory
    current_dir = Path(__file__).resolve().parent

    # List of files that usually live at the very root of a project
    root_markers = [".git", "setup.py", "pyproject.toml"]

    for parent in [current_dir, *current_dir.parents]:
        # If any of the marker files exist in this folder, then return the root
        if any((parent / marker).exists() for marker in root_markers):
            return str(parent)

    raise FileNotFoundError("Could not find project root folder.")
