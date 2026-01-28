import sys
import subprocess
import os

def open_file_externally(path: str):
    """
    Open a file or directory in the default OS application.
    Supports macOS, Windows, and Linux.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        return

    try:
        if sys.platform == "darwin":  # macOS
            subprocess.run(["open", path], check=True)
        elif sys.platform == "win32":  # Windows
            os.startfile(path)
        else:  # Linux / Unix
            subprocess.run(["xdg-open", path], check=True)
    except Exception as e:
        print(f"Failed to open '{path}': {e}", file=sys.stderr)
