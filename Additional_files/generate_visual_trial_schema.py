import random
import numpy as np
import pandas as pd

def create_schema_file(new_filename, initial_line, strings_list):
    # Create a DataFrame from the list
    df = pd.DataFrame(strings_list, columns=[initial_line])
    # Save the DataFrame to a CSV file
    df.to_csv(new_filename, index=False)

def shuffle_subarray(arr, positions):
    # Extract the elements to be shuffled
    subarray = [arr[pos] for pos in positions]
    # Shuffle the subarray
    random.shuffle(subarray)
    # Place the shuffled elements back into the original array
    for i, pos in enumerate(positions):
        arr[pos] = subarray[i]
    return arr

initial_line = "Length(min);30;Day;10;ExpGroup;1;Mouse;M2;DOB;20170402;Strain;PVCre1;Stop Threshold;-1000;Valve Open Time;0.9;Comments;no"
for j, training_day in enumerate(np.arange(1, 10)):

    np.random.seed(j+8)
    resets = np.append(np.array([20000]), np.repeat(250,119 * 50))  # first trial 20 seconds (20000ms) in darkness followed by 50 repeats of 119 images
    track_locations = np.repeat(np.arange(-1170, 10.5, 10), 50)
    np.random.shuffle(track_locations)
    while np.any(np.diff(track_locations) == 0):
        track_locations = shuffle_subarray(track_locations, positions=np.where(np.diff(track_locations) == 0)[0])
    track_locations = np.append(np.array([10]), track_locations)
    track_locations = np.array(track_locations, dtype=np.int64)
    my_strings = []
    for i in range(len(track_locations)):
        trial_tmp_string = ["track;", str(track_locations[i]),";rew_delay;0.0;gain modulation;1.0;rewarded;1;Reset;",str(resets[i]),
                            ";RZ start;8.8;RZ end;11.0;frequency;1;pulse_duration;100;max_duration;5;location_on;3.0;location_off;9.0;Valve_Open_Time;0.07"]
        trial_tmp_string = "".join(trial_tmp_string)
        my_strings.append(trial_tmp_string)
    create_schema_file(new_filename="/mnt/datastore/Harry/vr_scipts/blender_scripts/visual_task/flash_visual"+str(training_day)+".csv", initial_line=initial_line, strings_list=my_strings)
