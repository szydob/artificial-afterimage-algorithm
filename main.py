from pathlib import Path
import subprocess
import sys


def main() -> int:
    gui_path = Path(__file__).with_name("gui.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(gui_path)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
