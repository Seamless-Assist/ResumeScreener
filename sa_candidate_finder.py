import subprocess
import sys
from pathlib import Path
import os

# Resolve the root directory of the project
ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    # If Nixpacks runs `python -m sa_candidate_finder`,
    # it will find this file at the root and launch the Flask web app!
    script_path = ROOT / "web" / "app.py"
    print(f"Intercepted Nixpacks default start command. Redirecting to: {script_path}")
    
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except KeyboardInterrupt:
        pass
