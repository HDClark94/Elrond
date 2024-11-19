import pandas as pd
from Elrond.Helpers.upload_download import *

# Behavioural variables can be stored as an output from blender log files, these are made up of a csv-like format
# whereby rows indicate a timestep and columns represent some task variable.
# Time is not synced between other aquisitions like video or ephys.
# This script will convert the data within the blender file into
# a behavioural dataframe using .csv or .pkl file formats.

def generate_position_data_from_blender_file(recording_path, processed_path):
    blender_files = [f for f in Path(recording_path).iterdir() if "blender.csv" in f.name and f.is_file()]
    assert len(blender_files) == 1, "I need one blender.csv file, There isn't exactly one"

    blender_data = pd.read_csv(blender_files[0], skiprows=4, sep=";",
                               names=["Time", "Position-X", "Speed", "Speed/gain_mod", "Reward_received",
                                      "Reward_failed", "Lick_detected", "Tone_played","Tot trial", "gain_mod", 
                                      "rz_start", "rz_end", "sync_pulse", "curser_x", "curser_y", "speed_I_think"]) 
    blender_data.dropna()  
    position_data = pd.DataFrame()
    position_data["time_seconds"] = blender_data.index
    position_data["time_seconds"] = position_data["time_seconds"].values-np.min(position_data["time_seconds"]) # start at zero
    position_data["position_x"] = blender_data["curser_x"].values
    position_data["position_y"] = blender_data["curser_y"].values
    position_data["sync_pulse"] = blender_data["sync_pulse"].values 
    position_data["syncLED"] = blender_data["sync_pulse"].values 
    position_data["Reward_received"] = blender_data["Reward_received"].values
    save_path = processed_path + "position_data.csv"
    position_data.to_csv(save_path) 
    print("position data has been extracted from blender files saved at ", save_path)
    return position_data
