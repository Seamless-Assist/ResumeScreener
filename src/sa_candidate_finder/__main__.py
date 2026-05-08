import subprocess
import sys
from pathlib import Path

# Resolve the root directory of the project
ROOT = Path(__file__).resolve().parent.parent.parent

if __name__ == "__main__":
    # If Nixpacks forces `python -m sa_candidate_finder`,
    # we simply intercept it and launch the Flask web app instead!
    script_path = ROOT / "web" / "app.py"
    print(f"Intercepted Nixpacks default start command. Redirecting to: {script_path}")
    
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except KeyboardInterrupt:
        pass
