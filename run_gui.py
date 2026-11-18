"""Python entry point to launch the Streamlit House Price GUI in default browser."""

import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    app_path = ROOT / "app.py"
    if not app_path.exists():
        print(f"Error: Could not find {app_path}")
        sys.exit(1)

    print("=" * 60)
    print("🏡 Starting House Price Prediction ML - Web GUI")
    print("=" * 60)
    print("Launching Streamlit...")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",
        "--browser.gatherUsageStats=false"
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nGUI stopped by user.")

if __name__ == "__main__":
    main()
