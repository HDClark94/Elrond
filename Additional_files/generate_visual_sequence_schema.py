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

np.random.seed(0)

image_sequence = np.arange(-1170, 0.5, 10)
np.random.shuffle(image_sequence)

initial_loc = 10.0
initial_reset = 20000
final_reset =1000000
final_loc = 10.0

image_sequence = image_sequence[:50]

image_sequence = np.append(np.array([10]), image_sequence)
trial_resets   = np.append(np.array([5000]), np.repeat(250, 50))

my_strings = []

# add initial blank starter "trial"
trial_tmp_string = ["track;", str(initial_loc),";rew_delay;0.0;gain modulation;1.0;rewarded;1;Reset;",str(initial_reset),
                    ";RZ start;8.8;RZ end;11.0;frequency;1;pulse_duration;100;max_duration;5;location_on;3.0;location_off;9.0;Valve_Open_Time;0.07"]
trial_tmp_string = "".join(trial_tmp_string)
my_strings.append(trial_tmp_string)

# add 100 trials of a sequence of 50 images plus one 5 second blank phase
for i in range(100):
    for j in range(len(image_sequence)):
        trial_tmp_string = ["track;", str(image_sequence[j]),";rew_delay;0.0;gain modulation;1.0;rewarded;1;Reset;",str(trial_resets[j]),
                    ";RZ start;8.8;RZ end;11.0;frequency;1;pulse_duration;100;max_duration;5;location_on;3.0;location_off;9.0;Valve_Open_Time;0.07"]
        trial_tmp_string = "".join(trial_tmp_string)
        my_strings.append(trial_tmp_string)

# add final blank "trial"
trial_tmp_string = ["track;", str(final_loc),";rew_delay;0.0;gain modulation;1.0;rewarded;1;Reset;",str(final_reset),
                    ";RZ start;8.8;RZ end;11.0;frequency;1;pulse_duration;100;max_duration;5;location_on;3.0;location_off;9.0;Valve_Open_Time;0.07"]
trial_tmp_string = "".join(trial_tmp_string)
my_strings.append(trial_tmp_string)

create_schema_file(new_filename="/mnt/datastore/Harry/vr_scipts/blender_scripts/visual_task/flash_visual_sequence.csv", initial_line=initial_line, strings_list=my_strings)