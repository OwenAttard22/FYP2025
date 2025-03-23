import subprocess
import sys

def Select():
    """Opens BlueSky.py in a new PowerShell window and auto-closes it when done."""

    if sys.platform == "win32":
        # Windows: Open PowerShell, run Python script, then close the window
        subprocess.Popen(
            ["start", "powershell", "-NoExit", "-Command", "py ./Bluesky.py --discoverable --scenfile Alpha/TRAIN/train_0011.scn; exit"], 
            shell=True
        )

    print("✅ Alpha Scenario 11 Started in a new PowerShell window.")

