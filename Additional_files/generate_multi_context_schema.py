import numpy as np
import pandas as pd

familiar_context_beaconed_line = "track;0;rew_delay;0.0;gain modulation;1.0;rewarded;1;Reset;23;RZ start;8.8;RZ end;11.0;frequency;1;pulse_duration;100;max_duration;5;location_on;3.0;location_off;9.0;Valve_Open_Time;0.07"
familiar_context_nonbeaconed_line = "track;10;rew_delay;0.0;gain modulation;1.0;rewarded;1;Reset;23;RZ start;8.8;RZ end;11.0;frequency;1;pulse_duration;100;max_duration;5;location_on;3.0;location_off;9.0;Valve_Open_Time;0.07"
novel_context_beaconed_line = "track;20;rew_delay;0.0;gain modulation;1.0;rewarded;1;Reset;23;RZ start;11.8;RZ end;14.0;frequency;1;pulse_duration;100;max_duration;5;location_on;3.0;location_off;9.0;Valve_Open_Time;0.07"
novel_context_nonbeaconed_line = "track;30;rew_delay;0.0;gain modulation;1.0;rewarded;1;Reset;23;RZ start;11.8;RZ end;14.0;frequency;1;pulse_duration;100;max_duration;5;location_on;3.0;location_off;9.0;Valve_Open_Time;0.07"

familiar_trial_lines = [familiar_context_beaconed_line, familiar_context_nonbeaconed_line]
novel_trial_lines = [novel_context_beaconed_line, novel_context_nonbeaconed_line]

initial_line = "Length(min);30;Day;10;ExpGroup;1;Mouse;M2;DOB;20170402;Strain;PVCre1;Stop Threshold;3;Valve Open Time;0.9;Comments;no"
num_lines = 1000 # set n_lines to more trials than can be expected of a mouse in a 30 minute session

for training_day in np.arange(1, 50, 1):
    new_filename="/mnt/datastore/Harry/vr_scipts/blender_scripts/multicontext_trial_schemas_ratio_11_no_probe_pseudorandom/trial_schemas_ratio_11_no_probe_pseudorandom_training_day_"+str(training_day)+".csv"

    string_list = []
    for block in np.tile(["familiar", "novel"], 50):
        if block == "familiar":
            block_trial_list = []
            block_trial_list.append(familiar_context_beaconed_line) # always start with a beaconed trial
            block_trial_list.extend(np.random.choice([familiar_context_beaconed_line, familiar_context_nonbeaconed_line], p=[0.5, 0.5], size=19).tolist())
        elif block == "novel":
            block_trial_list = []
            block_trial_list.append(novel_context_beaconed_line) # always start with a beaconed trial
            block_trial_list.extend(np.random.choice([novel_context_beaconed_line, novel_context_nonbeaconed_line], p=[0.5, 0.5], size=19).tolist())
        string_list.extend(block_trial_list)
        
    # Create a DataFrame from the list
    df = pd.DataFrame(string_list, columns=[initial_line])

    # Save the DataFrame to a CSV file
    df.to_csv(new_filename, index=False)