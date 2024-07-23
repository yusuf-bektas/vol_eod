import json
import os
import subprocess
import sys

def run_script(script_path, args):
    """ Run a Python script with the given arguments """
    command = [sys.executable, script_path] + args
    print(f"Running command: {' '.join(command)}", flush=True)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}", flush=True)
        print(result.stderr, flush=True)
        raise RuntimeError(f"{script_path} failed with error code {result.returncode}")
    print(result.stdout, flush=True)
    print(result.stderr, flush=True)
    
def main():
    # Define the path to the main configuration file
    main_config_path = "./main.json"

    # Ensure the main.json exists
    if not os.path.isfile(main_config_path):
        raise FileNotFoundError("main.json not found!")

    # Read the main.json
    with open(main_config_path, 'r') as file:
        main_config = json.load(file)
    main_config_str = json.dumps(main_config)

    # Extract the model_config.json path
    model_config_path = main_config["model_cnf"]

    # Ensure the model_config.json path is valid
    if not os.path.isfile("." + model_config_path):
        raise FileNotFoundError(f"model_config.json not found at path: {model_config_path}")

    model_config_path = "./model_config.json"
    with open(model_config_path, 'r') as file:
        model_config = json.load(file)
    model_config_str = json.dumps(model_config)    

    for i in range(len(main_config["assets"])):
        print(f"Asset {i+1}: {main_config['assets'][i]['asset_mss']}", flush=True)
        print(f"Asset {i+1}: {main_config['assets'][i]['asset_tt']}", flush=True)
        asset_mss = main_config["assets"][i]["asset_mss"]
        asset_tt = main_config["assets"][i]["asset_tt"]
        freq = main_config["assets"][i]["freq"]

        # Run fetch_hist_data.py with asset_mss
        print("Running fetch_hist_data.py...", flush=True)
        fetch_hist_data_args = ["--asset", asset_mss, "--config", model_config_str]
        run_script("src/fetch_hist_data.py", fetch_hist_data_args)

        # Run vol_eod_model.py with asset_tt
        print("Running vol_eod_model.py...", flush=True)
        vol_eod_model_args = ["--asset", asset_mss, "--config", model_config_str, "--freq", freq]
        run_script("src/vol_eod_model.py", vol_eod_model_args)

        print("Running real time model output script...", flush=True)
        real_time_model_output_args = [
            "--assetTT", f"'{asset_tt}'",
            "--assetMSS", f"'{asset_mss}'",
            "--configMain", main_config_str,
            "--configModel", model_config_str
        ]
        #real_time_model_output_args = ["--assetTT", asset_tt, "--assetMSS", asset_mss, "--configMain", main_config_str, "--configModel", model_config_str]
        #run_script("src/model_update.py", real_time_model_output_args)
        from src.model_update import main
        main(asset_tt, asset_mss, main_config, model_config)
        print("All scripts ran successfully.", flush=True)

if __name__ == "__main__":
    main()
