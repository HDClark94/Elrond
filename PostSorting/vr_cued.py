import pandas as pd
import PostSorting.parameters
import os
import open_ephys_IO
import matplotlib.pyplot as plt
import numpy as np

def order_by_goal_location(processed_position_data):

    goal_locations = processed_position_data.goal_location
    trial_numbers = processed_position_data.goal_location_trial_numbers
    trial_types = processed_position_data.goal_location_trial_types

    tmp = np.array([goal_locations[~np.isnan(goal_locations)],
                    trial_numbers[~np.isnan(trial_numbers)],
                    trial_types[~np.isnan(trial_types)]])

    sortedtmp = tmp[:, tmp[0].argsort()]  # sorts by goal_location
    new_trial_numbers = np.arange(1, len(tmp[0]) + 1)

    del processed_position_data['goal_location_trial_numbers']

    processed_position_data['goal_location_new_trial_numbers'] = pd.Series(new_trial_numbers)
    processed_position_data['goal_location_old_trial_numbers'] = pd.Series(sortedtmp[1])

    # now swap trial numbers for binned_speed
    n_beaconed_trials = int(processed_position_data.beaconed_total_trial_number[0])
    n_nonbeaconed_trials = int(processed_position_data.nonbeaconed_total_trial_number[0])
    n_probe_trials = int(processed_position_data.probe_total_trial_number[0])

    n_total = n_beaconed_trials + n_nonbeaconed_trials + n_probe_trials

    for i in range(n_total):
        old_trial_number = processed_position_data['goal_location_old_trial_numbers'][i]
        new_trial_number = processed_position_data['goal_location_new_trial_numbers'][i]

        processed_position_data['speed_trial_numbers'][processed_position_data['speed_trial_numbers'] == old_trial_number] = new_trial_number
        processed_position_data['speed_trials_beaconed_trial_number'][processed_position_data['speed_trials_beaconed_trial_number'] == old_trial_number] = new_trial_number
        processed_position_data['speed_trials_non_beaconed_trial_number'][processed_position_data['speed_trials_non_beaconed_trial_number'] == old_trial_number] = new_trial_number

    return processed_position_data

def order_by_cue(beaconed, non_beaconed, probe, trial_bb_start, trial_bb_end):
    # in the process of replacing this function with order_by_goal
    '''
    :param beaconed: 2d np array, one stop/spike per row [stop/spike location, trial number, trial type]
    :param non_beaconed: ''
    :param probe: ''
    :param trial_bb_start: list of black box centres relative to goal location per trial
    :param trial_bb_end: ''
    :return: complete set of reordered inputs
    '''
    tmp = np.array([np.arange(1, len(trial_bb_start)+1),trial_bb_start, trial_bb_end])
    sortedtmp = tmp[:, tmp[1].argsort()] # sorts by blackbox centres

    trial_bb_start = list(sortedtmp[1])
    trial_bb_end = list(sortedtmp[2])

    sorted_trial_numbers = sortedtmp[0]
    new_trial_numbers = np.arange(1,len(trial_bb_start)+1)

    counter = 0
    for row in beaconed:
        if not math.isnan(row[1]):
            old_trial_number = int(row[1])
            new_trial_number = int(new_trial_numbers[sorted_trial_numbers == old_trial_number])
            row[1] = new_trial_number
            beaconed[counter] = row
        counter += 1

    counter = 0
    for row in non_beaconed:
        if not math.isnan(row[1]):
            old_trial_number = int(row[1])
            new_trial_number = int(new_trial_numbers[sorted_trial_numbers == old_trial_number])
            row[1] = new_trial_number
            non_beaconed[counter] = row
        counter += 1

    return beaconed, non_beaconed, probe, trial_bb_start, trial_bb_end


def add_goal_location(recording_to_process, raw_position_data, prm):
    raw_goal_data = extract_goal_locations(recording_to_process, prm)
    raw_position_data['in_goal_binary'] = np.asarray(raw_goal_data, dtype=np.float16)  # fill in dataframe
    raw_position_data = goal_binary2cm(raw_position_data, prm)
    raw_position_data = offset_location_by_goal(raw_position_data)
    return raw_position_data

def offset_location_by_goal(raw_position_data):
    raw_position_data["x_position_cm_offset"] = raw_position_data["goal_location_cm"] - raw_position_data["x_position_cm"]
    return raw_position_data

def extract_goal_locations(recording_folder, prm):
    goal_file_path = recording_folder + '/' + prm.get_goal_location_channel()

    if os.path.exists(goal_file_path):
        goal_location = open_ephys_IO.get_data_continuous(prm, goal_file_path)
    else:
        print('Movement or goal location data was not found.')
    if goal_location.shape[0] > 90000000:
        goal_location = goal_location[:90000000]

    goal_location = np.asarray(goal_location, dtype=np.float16)
    goal_location = np.append(np.array([0]), np.diff(goal_location))
    goal_location = np.absolute(goal_location)

    floor = np.round(min(goal_location), decimals=1)
    goal_location[np.round(goal_location, decimals=1) != floor] = 1
    goal_location[np.round(goal_location, decimals=1) == floor] = 0

    plot_goal_channel(goal_location, prm)
    return goal_location

def goal_binary2cm(raw_position_data, prm):
    '''
    translates the binary of being within the goal_location to a standard cm reading across all timesteps
    [000111000] -> [333333333]
    :param raw_position_data: pandas dataframe requiring in goal binary and trial number column
    :param prm: parameter class looks for
    :return:
    '''

    goal_location = np.array([])

    for trial_number in range(1, max(raw_position_data["trial_number"])+1):
        trial_raw_pos =  np.asarray(raw_position_data["x_position_cm"][raw_position_data["trial_number"] == trial_number])
        trial_raw_goal = np.asarray(raw_position_data["in_goal_binary"][raw_position_data["trial_number"] == trial_number])

        goal_locations_cm = trial_raw_pos[trial_raw_goal == 1.0]

        if len(goal_locations_cm)>0:
            goal_centre = (max(goal_locations_cm)+ min(goal_locations_cm))/2
        else:
            goal_centre = 0    # this catches if no goal location is detectable

        trial_goal_location = np.ones(len(trial_raw_pos))*goal_centre

        goal_location = np.append(goal_location, trial_goal_location)

    raw_position_data["goal_location_cm"] = list(goal_location)
    del raw_position_data["in_goal_binary"]

    return raw_position_data

def plot_goal_channel(goal_location, prm):
    save_path = prm.get_output_path() + '/Figures'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.plot(goal_location)
    plt.savefig(save_path + '/goal_location' + '.png')
    plt.close()

def test_goal_binary2cm(prm):

    print(" ------- unit test one, goal location constant -----------")

    actual_df = pd.DataFrame()
    actual_df["trial_number"] =  np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])  # len() = 16
    actual_df["x_position_cm"] = np.array([1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 1, 2, 3, 4, 4, 5])
    actual_df["in_goal_binary"] =np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0])  # 1 when x pos is 2 or 3 or 4
    actual_df = goal_binary2cm(actual_df, prm)

    expected_df = pd.DataFrame()
    expected_df["trial_number"] =  np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])   # len() = 16
    expected_df["x_position_cm"] = np.array([1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 1, 2, 3, 4, 4, 5])
    expected_df["in_goal_binary"] =np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0])  # 1 when x pos is 2 or 3 or 4
    expected_df["goal_location_cm"] = np.array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])

    if expected_df.equals(actual_df):
        print("goal_binary2cm passed unit test one")



    print(" ------- unit test two, goal location variable -----------")

    actual_df = pd.DataFrame()
    actual_df["trial_number"] =   np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])  # len() = 16
    actual_df["x_position_cm"] =  np.array([1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 1, 2, 3, 4, 4, 5])
    actual_df["in_goal_binary"] = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0])  # 1 when x pos is 2 or 3 or 4
    actual_df = goal_binary2cm(actual_df, prm)

    expected_df = pd.DataFrame()
    expected_df["trial_number"] =   np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])  # len() = 16
    expected_df["x_position_cm"] =  np.array([1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 1, 2, 3, 4, 4, 5])
    expected_df["in_goal_binary"] = np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0])  # 1 when x
    expected_df["goal_location_cm"] = np.array([4., 4., 4., 4., 4., 4., 3., 3., 3., 3., 3.5, 3.5, 3.5, 3.5, 3.5, 3.5])

    if expected_df.equals(actual_df):
        print("goal_binary2cm passed unit test two")


#  this is here for testing
def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    params = PostSorting.parameters.Parameters()

    test_goal_binary2cm(params)

if __name__ == '__main__':
    main()