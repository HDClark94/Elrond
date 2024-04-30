import random
import numpy as np
import pandas as pd

def create_schema_file(new_filename, initial_line, strings_list, num_lines):
    new_string_list = np.random.choice(strings_list, p=[0.5, 0.5], size=num_lines)

    # Create a DataFrame from the list
    df = pd.DataFrame(new_string_list, columns=[initial_line])

    # Save the DataFrame to a CSV file
    df.to_csv(new_filename, index=False)

# Example usage:
my_strings = ["track;0;rew_delay;0.0;gain modulation;1.0;rewarded;1;Reset;20;RZ start;8.8;RZ end;11.0;frequency;1;pulse_duration;100;max_duration;5;location_on;3.0;location_off;9.0;Valve_Open_Time;0.4",
              "track;10;rew_delay;0.0;gain modulation;1.0;rewarded;1;Reset;20;RZ start;8.8;RZ end;11.0;frequency;1;pulse_duration;100;max_duration;5;location_on;3.0;location_off;9.0;Valve_Open_Time;0.4"]
initial_line = "Length(min);30;Day;10;ExpGroup;1;Mouse;M2;DOB;20170402;Strain;PVCre1;Stop Threshold;3;Valve Open Time;0.9;Comments;no"
num_lines = 1000 # set n_lines to more trials than can be expected of a mouse in a 30 minute session

for training_day in np.arange(1, 50, 1):
    create_schema_file(new_filename="/mnt/datastore/Harry/vr_scipts/blender_scripts/trial_schemas_ratio_11_no_probe_pseudorandom/"
                                  "trial_schemas_ratio_11_no_probe_pseudorandom_training_day_"+str(training_day)+".csv",
                     initial_line=initial_line, strings_list=my_strings, num_lines=num_lines)
