import subprocess
import sys
import os
import random

def Select2(cf_type="TRAIN",cf_env="Alpha",count=1,debug=True):
     if sys.platform == "win32":
        # Windows: Open PowerShell, run Python script, then close the window
        bluesky_root = os.path.abspath(r"C:\Users\owena\FYP2025\FYP2025")
        env = os.environ.copy()
        env["PYTHONPATH"] = bluesky_root
        bs_main = os.path.join(bluesky_root, "bluesky", "__main__.py")
        
        
        if cf_type == "TRAIN":
            if cf_type == "TRAIN":
                random_number = random.randint(1, 1000)
                # random_number = 1001
            elif cf_type == "TEST":
                random_number = random.randint(1, 100)
            scen_file =  os.path.join(bluesky_root, "scenario", cf_env.upper(), cf_type.upper(), f"train_{random_number:04d}.scn")
        elif cf_type == "TEST":
            test_file = f"test_{count:04d}.scn"
            scen_file =  os.path.join(bluesky_root, "scenario", cf_env.upper(), cf_type.upper(), test_file)
        # scen_file =  os.path.join(bluesky_root, "scenario", cf_env.upper(), "TEST", "test_0001.scn")
            
        bluesky_cmd = [
            sys.executable, bs_main, "--scenfile", f"{scen_file}"
        ]
        
        subprocess.Popen(
            bluesky_cmd,
            env=env,
            cwd=bluesky_root,
            creationflags=subprocess.CREATE_NO_WINDOW if debug == False else 0
        )


    
    # """Opens BlueSky headlessly and ensures PYTHONPATH is set correctly."""
    # print("Slecting scenario..")

    # # Get absolute path to the folder containing the `bluesky` package
    # if sys.platform == "win32":
    #     bluesky_root = os.path.abspath(r"C:\Users\owena\FYP2025\FYP2025")
    # else:
    #     bluesky_root = os.path.abspath("/home/paperspace/FYP2025")
    
    # # Update the environment to include PYTHONPATH
    # env = os.environ.copy()
    # env["PYTHONPATH"] = bluesky_root

    # # Use __main__.py for direct control
    # bs_main = os.path.join(bluesky_root, "bluesky", "__main__.py")
    
    # if type == "TRAIN":
    #     random_number = random.randint(1, 1000)
    # elif type == "TEST":
    #     random_number = random.randint(1, 100)
    # else:
    #     raise ValueError("Invalid type. Use 'TRAIN' or 'TEST'.")
    
    # scenfile = os.path.join(bluesky_root, "scenario", "Alpha", "TRAIN", f"train_{random_number:04d}.scn")
    # # scenfile = os.path.join(bluesky_root, "scenario", "Alpha", "TRAIN", "train_0010.scn")

    # bluesky_cmd = [
    #     sys.executable, bs_main, "--scenfile", scenfile
    # ]

    # subprocess.Popen(
    #     bluesky_cmd,
    #     env=env,
    #     cwd=bluesky_root,
    #     creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    # )

# if __name__ == "__main__":
#     Select()


import subprocess
import sys
import os

def Select():
    """Starts BlueSky headlessly (without opening a new terminal window)."""
    
    bluesky_cmd = [
        sys.executable,  # Use the current Python interpreter
        "BlueSky.py",
        # "--headless",
        "--scenfile", "Alpha/TRAIN/train_0010.scn"
    ]

    if sys.platform == "win32":
        # Hide the window on Windows
        subprocess.Popen(
            bluesky_cmd,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
    else:
        # Standard subprocess for Unix
        subprocess.Popen(bluesky_cmd)

    print("✅ Alpha Scenario 11 started headlessly.")

if __name__ == "__main__":
    Select()
