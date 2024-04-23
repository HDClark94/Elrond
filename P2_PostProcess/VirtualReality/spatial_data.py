import numpy as np
import pandas as pd
import os
import traceback
from scipy import stats
import settings
from astropy.convolution import convolve, Gaussian1DKernel
import Helpers.metadata_extraction as metadata_extraction
import matplotlib.pylab as plt
from neuroconv.utils.dict import load_dict_from_file


def get_track_length(recording_path):
    if os.path.exists(recording_path+"/params.yml"):
        params = load_dict_from_file(recording_path + "/params.yml")
        return params["track_length"]
    elif os.path.exists(recording_path+"/parameters.txt"):
        parameter_file_path = metadata_extraction.get_tags_parameter_file(recording_path)
        _, track_length = metadata_extraction.process_running_parameter_tag(parameter_file_path)
        return track_length
    else:
        raise AssertionError("I could not find a params.yml or parameters.txt file")


def get_stop_threshold(recording_path):
    if os.path.exists(recording_path+"/params.yml"):
        params = load_dict_from_file(recording_path + "/params.yml")
        return params["stop_threshold"]
    elif os.path.exists(recording_path+"/parameters.txt"):
        parameter_file_path = metadata_extraction.get_tags_parameter_file(recording_path)
        stop_threshold, _ = metadata_extraction.process_running_parameter_tag(parameter_file_path)
        return stop_threshold
    else:
        raise AssertionError("I could not find a params.yml or parameters.txt file")


def add_speed_per_100ms(position_data, track_length):
    sampling_rate = (np.abs(position_data["time_seconds"] - 1)).argmin() # dedeuce sampling rate
    positions = np.array(position_data["x_position_cm"])
    trial_numbers = np.array(position_data["trial_number"])

    distance_travelled = positions + (track_length * (trial_numbers - 1))
    change_in_distance_travelled = np.concatenate([np.zeros(1), np.diff(distance_travelled)], axis=0)
    n_samples_per_100ms = int(sampling_rate * 0.1)  # 0.1 seconds == 100ms
    speed_per_100ms = np.array(pd.Series(change_in_distance_travelled).rolling(n_samples_per_100ms).sum()) / 0.1  # 0.1 seconds == 100ms

    # we can expect nan values for the first 100ms
    # replace these with the first non nan value
    speed_per_100ms[np.isnan(speed_per_100ms)] = speed_per_100ms[np.isfinite(speed_per_100ms)][0]

    position_data["speed_per_100ms"] = speed_per_100ms
    return position_data


def add_stopped_in_rz(position_data, track_length, stop_threshold):
    # TODO track features should be inherited from a parameter file
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30
    track_start = 30
    track_end = track_length-30

    position_data["below_speed_threshold"] = position_data["speed_per_100ms"] < stop_threshold
    position_data["stopped_in_rz"] = (position_data["below_speed_threshold"] == True) &\
                                     (position_data["x_position_cm"] <= reward_zone_end) & \
                                     (position_data["x_position_cm"] >= reward_zone_start)
    return position_data


def add_hit_according_to_blender(position_data, processed_position_data):
    hit_trial_numbers = np.unique(position_data[position_data["stopped_in_rz"] == True]["trial_number"])
    hit_array = np.zeros(len(processed_position_data), dtype=int)
    hit_array[hit_trial_numbers-1] = 1
    processed_position_data["hit_blender"] = hit_array
    return processed_position_data


def add_stops_according_to_blender(position_data, processed_position_data):
    stop_locations = []
    first_stop_locations = []
    for tn in processed_position_data["trial_number"]:
        trial_stop_locations = np.array(position_data[(position_data["below_speed_threshold"] == True)
                                                      & (position_data["trial_number"] == tn)]['x_position_cm'])
        if len(trial_stop_locations)>0:
            trial_first_stop_location = trial_stop_locations[0]
        else:
            trial_first_stop_location = np.nan

        stop_locations.append(trial_stop_locations.tolist())
        first_stop_locations.append(trial_first_stop_location)

    processed_position_data["stop_location_cm"] = stop_locations
    processed_position_data["first_stop_location_cm"] = first_stop_locations
    return processed_position_data


def add_trial_variables(raw_position_data, processed_position_data, track_length):
    n_trials = max(raw_position_data["trial_number"])

    trial_numbers = []
    trial_types = []
    position_bin_centres = []
    for trial_number in range(1, n_trials+1):
        trial_type = int(stats.mode(np.array(raw_position_data['trial_type'][np.array(raw_position_data['trial_number']) == trial_number]), axis=None)[0])
        bins = np.arange(0, track_length+1, settings.vr_bin_size_cm)
        bin_centres = 0.5*(bins[1:]+bins[:-1])

        trial_numbers.append(trial_number)
        trial_types.append(trial_type)
        position_bin_centres.append(bin_centres)

    processed_position_data["trial_number"] = trial_numbers
    processed_position_data["trial_type"] = trial_types
    processed_position_data["position_bin_centres"] = position_bin_centres
    return processed_position_data


def bin_in_space(position_data, processed_position_data, track_length, smoothen=True):
    if smoothen:
        suffix="_smoothed"
    else:
        suffix=""

    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_space_cm/settings.vr_bin_size_cm)
    n_trials = max(processed_position_data["trial_number"])

    # extract spatial variables from position
    speeds = np.array(position_data['speed_per_100ms'], dtype="float64")
    times = np.array(position_data['time_seconds'], dtype="float64")
    trial_numbers = np.array(position_data['trial_number'], dtype=np.int64)
    x_position_cm = np.array(position_data['x_position_cm'], dtype="float64")
    x_position_elapsed_cm = (track_length*(trial_numbers-1))+x_position_cm

    # calculate the average speed and position in each 1cm spatial bin
    spatial_bins = np.arange(0, (n_trials*track_length)+1, settings.vr_bin_size_cm) # 1 cm bins
    speed_space_bin_means = (np.histogram(x_position_elapsed_cm, spatial_bins, weights = speeds)[0] / np.histogram(x_position_elapsed_cm, spatial_bins)[0])
    pos_space_bin_means = (np.histogram(x_position_elapsed_cm, spatial_bins, weights = x_position_elapsed_cm)[0] / np.histogram(x_position_elapsed_cm, spatial_bins)[0])
    tn_space_bin_means = (((0.5*(spatial_bins[1:]+spatial_bins[:-1]))//track_length)+1).astype(np.int64) # uncomment to get nan values for portions of first and last trial

    # and smooth
    if smoothen:
        speed_space_bin_means = convolve(speed_space_bin_means, gauss_kernel, boundary="extend")
        pos_space_bin_means = convolve(pos_space_bin_means, gauss_kernel, boundary="extend")

    # calculate the acceleration from the smoothed speed
    acceleration_space_bin_means = np.diff(np.array(speed_space_bin_means))
    acceleration_space_bin_means = np.hstack((0, acceleration_space_bin_means))

    # recalculate the position from the elapsed distance
    pos_space_bin_means = pos_space_bin_means%track_length

    # create empty lists to be filled and put into processed_position_data
    speeds_binned_in_space = []; pos_binned_in_space = []; acc_binned_in_space = []

    for trial_number in range(1, n_trials+1):
        speeds_binned_in_space.append(speed_space_bin_means[tn_space_bin_means == trial_number].tolist())
        pos_binned_in_space.append(pos_space_bin_means[tn_space_bin_means == trial_number].tolist())
        acc_binned_in_space.append(acceleration_space_bin_means[tn_space_bin_means == trial_number].tolist())

    processed_position_data["speeds_binned_in_space"+suffix] = speeds_binned_in_space
    processed_position_data["pos_binned_in_space"+suffix] = pos_binned_in_space
    processed_position_data["acc_binned_in_space"+suffix] = acc_binned_in_space

    return processed_position_data


def bin_in_time(position_data, processed_position_data, track_length, smoothen=True):
    if smoothen:
        suffix="_smoothed"
    else:
        suffix=""

    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_time_seconds/settings.time_bin_size)
    n_trials = max(position_data["trial_number"])

    # extract spatial variables from position
    speeds = np.array(position_data['speed_per_100ms'], dtype="float64")
    times = np.array(position_data['time_seconds'], dtype="float64")
    trial_numbers = np.array(position_data['trial_number'], dtype=np.int64)
    x_position_cm = np.array(position_data['x_position_cm'], dtype="float64")
    x_position_elapsed_cm = (track_length*(trial_numbers-1))+x_position_cm

    # calculate the average speed and position in each 100ms time bin
    time_bins = np.arange(min(times), max(times), settings.time_bin_size) # 100ms time bins
    speed_time_bin_means = (np.histogram(times, time_bins, weights = speeds)[0] / np.histogram(times, time_bins)[0])
    pos_time_bin_means = (np.histogram(times, time_bins, weights = x_position_elapsed_cm)[0] / np.histogram(times, time_bins)[0])
    tn_time_bin_means = (np.histogram(times, time_bins, weights = trial_numbers)[0] / np.histogram(times, time_bins)[0]).astype(np.int64)

    # and smooth
    if smoothen:
        speed_time_bin_means = convolve(speed_time_bin_means, gauss_kernel, boundary="extend")
        pos_time_bin_means = convolve(pos_time_bin_means, gauss_kernel, boundary="extend")

    # calculate the acceleration from the smoothed speed
    acceleration_time_bin_means = np.diff(np.array(speed_time_bin_means))
    acceleration_time_bin_means = np.hstack((0, acceleration_time_bin_means))

    # recalculate the position from the elapsed distance
    pos_time_bin_means = pos_time_bin_means%track_length

    # create empty lists to be filled and put into processed_position_data
    speeds_binned_in_time = []; pos_binned_in_time = []; acc_binned_in_time = []

    for trial_number in range(1, n_trials+1):
        speeds_binned_in_time.append(speed_time_bin_means[tn_time_bin_means == trial_number].tolist())
        pos_binned_in_time.append(pos_time_bin_means[tn_time_bin_means == trial_number].tolist())
        acc_binned_in_time.append(acceleration_time_bin_means[tn_time_bin_means == trial_number].tolist())

    processed_position_data["speeds_binned_in_time"+suffix] = speeds_binned_in_time
    processed_position_data["pos_binned_in_time"+suffix] = pos_binned_in_time
    processed_position_data["acc_binned_in_time"+suffix] = acc_binned_in_time
    return processed_position_data


def add_hit_try_run(processed_position_data):
    # hit try and run classification are explained in Clark and Nolan 2024 eLife

    # first get the avg speeds in the reward zone for all hit trials
    rewarded_processed_position_data = processed_position_data[processed_position_data["hit_blender"] == True]
    speeds_in_rz = np.array(rewarded_processed_position_data["avg_speed_in_RZ"])

    mean, sigma = np.nanmean(speeds_in_rz), np.nanstd(speeds_in_rz)
    interval = stats.norm.interval(0.95, loc=mean, scale=sigma)
    upper = interval[1]
    lower = interval[0]

    hit_miss_try =[]
    for i, trial_number in enumerate(processed_position_data.trial_number):
        trial_process_position_data = processed_position_data[(processed_position_data.trial_number == trial_number)]
        track_speed = trial_process_position_data["avg_speed_on_track"].iloc[0]
        speed_in_rz = trial_process_position_data["avg_speed_in_RZ"].iloc[0]

        if (trial_process_position_data["hit_blender"].iloc[0] == True) and (track_speed>settings.hit_try_run_speed_threshold):
            hit_miss_try.append("hit")
        elif (speed_in_rz >= lower) and (speed_in_rz <= upper) and (track_speed>settings.hit_try_run_speed_threshold):
            hit_miss_try.append("try")
        elif (speed_in_rz < lower) or (speed_in_rz > upper) and (track_speed>settings.hit_try_run_speed_threshold):
            hit_miss_try.append("run")
        else:
            hit_miss_try.append("rejected")

    processed_position_data["hit_miss_try"] = hit_miss_try
    return processed_position_data, upper


def add_avg_track_speed(position_data, processed_position_data, track_length):
    reward_zone_start = track_length-60-30-20
    reward_zone_end = track_length-60-30
    track_start = 30
    track_end = track_length-30

    avg_speed_on_tracks = []
    avg_speed_in_RZs = []
    for i, trial_number in enumerate(processed_position_data.trial_number):
        trial_processed_position_data = processed_position_data[processed_position_data["trial_number"] == trial_number]
        trial_position_data = position_data[position_data["trial_number"] == trial_number]
        speeds_in_time = np.array(trial_position_data["speed_per_100ms"])
        pos_in_time = np.array(trial_position_data["x_position_cm"])
        in_rz_mask = (pos_in_time > reward_zone_start) & (pos_in_time <= reward_zone_end)
        speeds_in_time_outside_RZ = speeds_in_time[~in_rz_mask]
        speeds_in_time_inside_RZ = speeds_in_time[in_rz_mask]

        if len(speeds_in_time_outside_RZ)==0:
            avg_speed_on_track = np.nan
        else:
            avg_speed_on_track = np.nanmean(speeds_in_time_outside_RZ)

        if len(speeds_in_time_inside_RZ) == 0:
            avg_speed_in_RZ = np.nan
        else:
            avg_speed_in_RZ= np.nanmean(speeds_in_time_inside_RZ)

        avg_speed_on_tracks.append(avg_speed_on_track)
        avg_speed_in_RZs.append(avg_speed_in_RZ)

    processed_position_data["avg_speed_on_track"] = avg_speed_on_tracks
    processed_position_data["avg_speed_in_RZ"] = avg_speed_in_RZs
    return processed_position_data


def process_position_data(position_data, track_length, stop_threshold):
    position_data = add_speed_per_100ms(position_data, track_length)
    position_data = add_stopped_in_rz(position_data, track_length, stop_threshold)

    processed_position_data = pd.DataFrame() # make dataframe for processed position data
    processed_position_data = add_trial_variables(position_data, processed_position_data, track_length)
    processed_position_data = bin_in_time(position_data, processed_position_data, track_length, smoothen=True)
    processed_position_data = bin_in_time(position_data, processed_position_data, track_length, smoothen=False)
    processed_position_data = bin_in_space(position_data, processed_position_data, track_length, smoothen=True)
    processed_position_data = bin_in_space(position_data, processed_position_data, track_length, smoothen=False)
    processed_position_data = add_hit_according_to_blender(position_data, processed_position_data)
    processed_position_data = add_stops_according_to_blender(position_data, processed_position_data)
    processed_position_data = add_avg_track_speed(position_data, processed_position_data, track_length)
    processed_position_data, _ = add_hit_try_run(processed_position_data)
    return processed_position_data


#  for testing
def main():
    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()
