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
    assert get_recording_types([recording_path])[0] == "vr", "recording type must be vr if attempting to generate position data from a blender file"

    blender_data = pd.read_csv(blender_files[0], skiprows=4, sep=";",
                               names=["Time", "Position-X", "Speed", "Speed/gain_mod", "Reward_received",
                                      "Reward_failed", "Lick_detected", "Tone_played", "Position-Y",
                                      "Tot trial", "gain_mod", "rz_start", "rz_end", "sync_pulse"])
    blender_data.dropna()
    position_data = pd.DataFrame()
    position_data["x_position_cm"] = blender_data["Position-X"]*10 # assumes 10 cm per virtual unit
    position_data["trial_number"] = blender_data["Tot trial"]
    position_data["trial_type"] = blender_position_to_trial_type(blender_data["Position-Y"])
    position_data["time_seconds"] = blender_data["Time"]
    position_data["speed_as_read_by_blender"] = blender_data["Speed"]
    position_data["Reward_received_as_read_by_blender"] = blender_data["Reward_received"]
    position_data["Tone_played_as_read_by_blender"] = blender_data["Tone_played"]
    position_data["sync_pulse"] = blender_data["sync_pulse"]
    position_data["dwell_time_ms"] = np.append(0, np.diff(position_data["time_seconds"]))

    save_path = processed_path + "position_data.csv"
    position_data.to_csv(save_path)
    print("position data has been extracted from blender files saved at ", save_path)
    return position_data

def blender_position_to_trial_type(blender_position):
    trial_types = []
    for i in range(len(blender_position)):
        if int(np.round(blender_position[i], 2)) == 0:
            trial_types.append(0)
        if int(np.round(blender_position[i], 2)) == 10:
            trial_types.append(1)
        if int(np.round(blender_position[i], 2)) == 20:
            trial_types.append(2)
    return np.array(trial_types)
