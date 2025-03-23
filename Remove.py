import subprocess
import sys

def Select(**kwargs):
    """Opens BlueSky.py in a new PowerShell window and auto-closes it when done."""
    
    scenario = kwargs.get("scenario", "Alpha")
    type = kwargs.get("type", "TRAIN")
    random = kwargs.get("random", False)
    print(f"Scenario: {scenario}, Type: {type}, Random: {random}")

    if sys.platform == "win32":
        # Windows: Open PowerShell, run Python script, then close the window
        subprocess.Popen(
            ["start", "powershell", "-NoExit", "-Command", "py ./Bluesky.py --discoverable --scenfile Alpha/TRAIN/train_0011.scn; exit"], 
            shell=True
        )

    print("✅ Alpha Scenario 11 Started in a new PowerShell window.")


if __name__ == "__main__":
    Select()