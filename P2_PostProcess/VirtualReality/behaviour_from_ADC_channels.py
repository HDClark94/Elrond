import os
import numpy as np
import pandas as pd
from scipy import stats
import Elrond.settings as settings
import Helpers.open_ephys_IO as open_ephys_IO
from astropy.convolution import convolve, Gaussian1DKernel
from Elrond.Helpers.upload_download import *

from Elrond.P2_PostProcess.VirtualReality.spatial_data import process_position_data, get_stop_threshold, get_track_length
from Elrond.P2_PostProcess.VirtualReality.plotting import plot_variables, plot_behaviour

# Behavioural variables in VR recordings created between 2016-2023
# are stored within the ADC*.continuous open ephys legacy files
# This was achieved by coding several important behavioural variables
# acquired by blender at 30 Hz, and reading them through the an
# open ephys IO board and into thee aquisition board at 30 kHz
# This afforded the reconstruction of behavioural data already time synced.
# However this method is also at the mercy of low hum noise of the ADC
# channel reading.

# This script will convert these ADC "behavioural" .continous files into
# a behavioural dataframe using .csv or .pkl file formats.


def correct_for_restart(location):
    cummulative_minimums = np.minimum.accumulate(location)
    local_min_median = np.median(cummulative_minimums)

    location [location <local_min_median] = local_min_median # deals with if the VR is switched off during recording - location value drops to 0 - min is usually 0.56 approx
    return location


def get_raw_location(recording_folder):
    print('Extracting raw location...')
    file_path = recording_folder + '/' + settings.movement_channel
    if os.path.exists(file_path):
        location = open_ephys_IO.get_data_continuous(file_path)
    else:
        raise FileNotFoundError('Movement data was not found.')
    location=correct_for_restart(location)
    return np.asarray(location, dtype=np.float16)


def calculate_track_location(position_data, recording_folder, track_length):
    recorded_location = get_raw_location(recording_folder) # get raw location from DAQ pin
    print('Converting raw location input to cm...')

    recorded_startpoint = min(recorded_location)
    recorded_endpoint = max(recorded_location)
    recorded_track_length = recorded_endpoint - recorded_startpoint
    distance_unit = recorded_track_length/track_length  # Obtain distance unit (cm) by dividing recorded track length to actual track length
    location_in_cm = (recorded_location - recorded_startpoint) / distance_unit

    position_data['x_position_cm'] = np.asarray(location_in_cm, dtype=np.float32) # fill in dataframe
    return position_data


def recalculate_track_location(position_data, track_length):
    # address the noise at the individual trial level
    recorded_location = np.asarray(position_data["x_position_cm"])
    trial_numbers = np.asarray(position_data["trial_number"])

    location_in_cm = np.array([])
    global_recorded_startpoint = min(recorded_location)
    global_recorded_endpoint = max(recorded_location)
    unique_trial_numbers =  np.unique(position_data["trial_number"])
    for tn in unique_trial_numbers:
        trial_locations = recorded_location[trial_numbers == tn]

        if tn == unique_trial_numbers[0]:
            recorded_startpoint = global_recorded_startpoint
            recorded_endpoint = max(trial_locations)
        elif tn == unique_trial_numbers[-1]:
            recorded_startpoint = min(trial_locations)
            recorded_endpoint = global_recorded_endpoint
        else:
            recorded_startpoint = min(trial_locations)
            recorded_endpoint = max(trial_locations)
        recorded_track_length = recorded_endpoint - recorded_startpoint
        distance_unit = recorded_track_length/track_length  # Obtain distance unit (cm) by dividing recorded track length to actual track length
        trial_location_in_cm = (trial_locations - recorded_startpoint) / distance_unit

        location_in_cm = np.concatenate((location_in_cm, trial_location_in_cm))
    position_data['x_position_cm'] = np.asarray(location_in_cm, dtype=np.float32) # fill in dataframe
    return position_data

def curate_track_location(raw_position_data, track_length):
    # do not allow trials to go back on themselves
    # trial(n) cannot be lower than trial(n-1)
    x_position_cm = np.array(raw_position_data["x_position_cm"], dtype=np.float64)
    trial_numbers = np.array(raw_position_data["trial_number"], dtype=np.int64)
    distance_travelled = x_position_cm + (track_length * (trial_numbers - 1))

    # nan out sections where trial number decreases
    bad_indices = ((np.diff(trial_numbers) >= 0) == False).nonzero()[0]
    for bad_idx in bad_indices:
        last_good_dt = distance_travelled[bad_idx]

        from_bad_idx_mask = np.zeros(len(distance_travelled))
        from_bad_idx_mask[bad_idx+1:] = 1
        from_bad_idx_mask = from_bad_idx_mask==1
        bad_tn_mask = trial_numbers == trial_numbers[bad_idx+1]
        mask = from_bad_idx_mask & bad_tn_mask

        distance_travelled[mask] = last_good_dt

    # remake trial numbers and position
    distance_travelled = distance_travelled-\
                         (distance_travelled[0] // track_length)*track_length # added redundancy so trials start at 1
    x_position_cm = distance_travelled % track_length
    trial_numbers = np.array(distance_travelled//track_length, dtype=np.int64)+1

    # trial numbers should start at 1
    assert trial_numbers[0] == 1
    # trial numbers should never go down
    assert np.all(np.diff(trial_numbers) >= 0)

    print('This mouse did ', len(np.unique(trial_numbers)), ' trials')
    raw_position_data["x_position_cm"] = x_position_cm
    raw_position_data["trial_number"] = trial_numbers
    return raw_position_data




def fix_around_teleports(raw_position_data, track_length):
    x_position_cm = np.array(raw_position_data["x_position_cm"], dtype=np.float64)
    trial_numbers = np.array(raw_position_data["trial_number"], dtype=np.int64)
    distance_travelled = x_position_cm + (track_length * (trial_numbers - 1))

    n_samples_to_nan_out = int(settings.sampling_rate * 0.2)  # 200 ms

    # add nans to inconcievable speeds
    change_in_distance_travelled = np.concatenate([np.zeros(1), np.diff(distance_travelled)], axis=0)
    bad_indices = np.asarray(abs(change_in_distance_travelled) > 10).nonzero()[0]
    for bad_idx in bad_indices:
        if bad_idx-n_samples_to_nan_out <= 0:
            bad_ind_from = 0
        else:
            bad_ind_from = bad_idx-n_samples_to_nan_out
        if bad_idx+n_samples_to_nan_out >= len(distance_travelled):
            bad_ind_to = len(distance_travelled)
        else:
            bad_ind_to = bad_idx+n_samples_to_nan_out

        assert (bad_ind_from >= 0) and (bad_ind_to <= len(distance_travelled))
        distance_travelled[bad_ind_from: bad_ind_to] = np.nan

    #now interpolate where these nan values are
    ok = ~np.isnan(distance_travelled)
    xp = ok.ravel().nonzero()[0]
    fp = distance_travelled[~np.isnan(distance_travelled)]
    x  = np.isnan(distance_travelled).ravel().nonzero()[0]
    distance_travelled[np.isnan(distance_travelled)] = np.interp(x, xp, fp)

    # remake trial numbers
    x_position_cm = distance_travelled % track_length
    trial_numbers = np.array(distance_travelled // track_length, dtype=np.int64) + 1

    raw_position_data["x_position_cm"] = x_position_cm
    raw_position_data["trial_number"] = trial_numbers
    return raw_position_data


def smoothen_track_location(raw_position_data, track_length):
    x_position_cm = np.array(raw_position_data["x_position_cm"], dtype=np.float64)
    trial_numbers = np.array(raw_position_data["trial_number"], dtype=np.int64)
    distance_travelled = x_position_cm + (track_length * (trial_numbers - 1))

    # smooth out the ephys noise
    gauss_kernel = Gaussian1DKernel(stddev=200)
    distance_travelled = convolve(distance_travelled, gauss_kernel, boundary="extend")

    x_position_cm = distance_travelled%track_length
    trial_numbers = np.array(distance_travelled//track_length, dtype=np.int64)+1

    raw_position_data["x_position_cm"] = x_position_cm
    raw_position_data["trial_number"] = trial_numbers
    return raw_position_data


def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]


def calculate_trial_numbers(position_data):
    print('Calculating trial numbers...')
    trials = np.zeros((position_data.shape[0]))
    new_trial_indices = get_new_trial_indices(position_data)
    trials = fill_in_trial_array(new_trial_indices,trials)

    position_data['trial_number'] = np.asarray(trials, dtype=np.uint16)
    return position_data


def get_new_trial_indices(position_data):
    location_diff = position_data['x_position_cm'].diff()  # Get the raw location from the movement channel
    trial_indices = np.where(location_diff < -20)[0] # return indices where is new trial
    trial_indices = check_for_trial_restarts(trial_indices) # check if trial_indices values are within some margin of eachother, if so, delete
    new_trial_indices = np.hstack((0,trial_indices,len(location_diff))) # add start and end indices so fills in whole arrays
    return new_trial_indices


def check_for_trial_restarts(trial_indices):
    new_trial_indices=[]
    for icount,i in enumerate(range(len(trial_indices)-1)):
        index_difference = trial_indices[icount] - trial_indices[icount+1]
        if index_difference > - settings.sampling_rate/2:
            continue
        else:
            index = trial_indices[icount]
            new_trial_indices = np.append(new_trial_indices,index)
    return new_trial_indices


def fill_in_trial_array(new_trial_indices,trials):
    trial_count = 1
    for icount,i in enumerate(range(len(new_trial_indices)-1)):
        new_trial_index = int(new_trial_indices[icount])
        next_trial_index = int(new_trial_indices[icount+1])
        trials[new_trial_index:next_trial_index] = trial_count
        trial_count += 1
    return trials


def calculate_trial_types(raw_position_data, recording_folder):
    print('Loading trial types from continuous...')
    first_ch = load_first_trial_channel(recording_folder)
    second_ch = load_second_trial_channel(recording_folder)

    trial_numbers = np.array(raw_position_data["trial_number"])
    trial_type = np.zeros(len(trial_numbers))
    trial_type[:] = np.nan

    print('Calculating trial type...')
    for tn in np.unique(trial_numbers):
        second = stats.mode(second_ch[trial_numbers == tn])[0][0]
        first = stats.mode(first_ch[trial_numbers == tn])[0][0]
        if second < 2 and first < 2: # if beaconed
            trial_type[trial_numbers == tn] = 0
        if second > 2 and first < 2: # if non beaconed
            trial_type[trial_numbers == tn] = 1
        if second > 2 and first > 2: # if probe
            trial_type[trial_numbers == tn] = 2
    raw_position_data['trial_type'] = np.asarray(trial_type, dtype=np.uint8)
    return raw_position_data


# two continuous channels represent trial type
def load_first_trial_channel(recording_folder):
    file_path = recording_folder + '/' + settings.first_trial_channel
    trial_first = open_ephys_IO.get_data_continuous(file_path)
    return np.asarray(trial_first, dtype=np.uint8)


# two continuous channels represent trial type
def load_second_trial_channel(recording_folder):
    file_path = recording_folder + '/' + settings.second_trial_channel
    trial_second = open_ephys_IO.get_data_continuous(file_path)
    return np.asarray(trial_second, dtype=np.uint8)


# calculate time from start of recording in seconds for each sampling point
def calculate_time(position_data, sampling_rate):
    print('Calculating time...')
    position_data['time_seconds'] = position_data['trial_number'].index/sampling_rate
    return position_data


def downsample_position_data(raw_position_data,
                             sampling_rate = settings.sampling_rate,
                             down_sampled_rate = settings.down_sampled_rate):
    position_data = pd.DataFrame()
    downsample_factor = int(sampling_rate/ down_sampled_rate)
    for column in list(raw_position_data):
        position_data[column] = raw_position_data[column][::downsample_factor]
    return position_data


def calculate_track_location(position_data, recording_folder, track_length):
    recorded_location = get_raw_location(recording_folder) # get raw location from DAQ pin
    print('Converting raw location input to cm...')

    recorded_startpoint = min(recorded_location)
    recorded_endpoint = max(recorded_location)
    recorded_track_length = recorded_endpoint - recorded_startpoint
    distance_unit = recorded_track_length/track_length  # Obtain distance unit (cm) by dividing recorded track length to actual track length
    location_in_cm = (recorded_location - recorded_startpoint) / distance_unit
    position_data['x_position_cm'] = np.asarray(location_in_cm, dtype=np.float16) # fill in dataframe
    return position_data


def extract_position_data(recording_path, track_length):
    raw_position_data = pd.DataFrame()
    raw_position_data = calculate_track_location(raw_position_data, recording_path, track_length)
    raw_position_data = calculate_trial_numbers(raw_position_data)
    raw_position_data = recalculate_track_location(raw_position_data, track_length)
    raw_position_data = fix_around_teleports(raw_position_data, track_length)
    raw_position_data = smoothen_track_location(raw_position_data, track_length)
    raw_position_data = curate_track_location(raw_position_data, track_length)
    raw_position_data = calculate_trial_types(raw_position_data, recording_path)
    raw_position_data = calculate_time(raw_position_data, sampling_rate=settings.sampling_rate)
    down_sampled_position_data = downsample_position_data(raw_position_data)
    return raw_position_data, down_sampled_position_data


def generate_position_data_from_ADC_channels(recording_path, processed_path):
    print("I will attempt to use ADC channel information")

    # recording type needs to be vr
    assert get_recording_types([recording_path])[0] == "vr"

    track_length = get_track_length(recording_path)
    stop_threshold = get_stop_threshold(recording_path)

    # extract and process position data
    raw_position_data, downsampled_position_data = extract_position_data(recording_path, track_length)
    save_path = processed_path + "position_data.csv"
    downsampled_position_data.to_csv(save_path)
    print("downsampled position data has been extracted from ADC channels and saved at ", save_path)
    return downsampled_position_data


def run_checks_for_position_data(position_data, recording_path, processed_folder_name):
    track_length = get_track_length(recording_path)
    stop_threshold = get_stop_threshold(recording_path)
    processed_position_data = process_position_data(position_data, track_length, stop_threshold)

    # make some plots
    output_path = recording_path+"/"+processed_folder_name
    plot_variables(position_data, output_path=output_path+"/Figures/Behaviour")
    plot_behaviour(processed_position_data, output_path, track_length)
    print("some plots have been saved at ", output_path, "")


def main():
    print("")
    print("")

if __name__ == '__main__':
    main()

