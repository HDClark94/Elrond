import PostSorting.parameters
import numpy as np
import pandas as pd
import settings
from astropy.convolution import convolve, Gaussian1DKernel
import control_sorting_analysis
import PostSorting.vr_sync_spatial_data
import traceback
import os
import sys

prm = PostSorting.parameters.Parameters()


def get_stop_threshold_and_track_length(recording_path):
    parameter_file_path = control_sorting_analysis.get_tags_parameter_file(recording_path)
    stop_threshold, track_length, _ = PostSorting.post_process_sorted_data_vr.process_running_parameter_tag(parameter_file_path)
    return stop_threshold, track_length

def add_speed(spike_data, raw_position_data):
    speed_per200ms = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_firing_indices = np.asarray(spike_data[spike_data.cluster_id == cluster_id].firing_times)[0]
        speed_per200ms.append(raw_position_data["speed_per200ms"][cluster_firing_indices].to_list())

    spike_data["speed_per200ms"] = speed_per200ms
    return spike_data


def add_position_x(spike_data, raw_position_data):
    x_position_cm = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_firing_indices = np.asarray(spike_data[spike_data.cluster_id == cluster_id].firing_times)[0]
        x_position_cm.append(raw_position_data["x_position_cm"][cluster_firing_indices].to_list())

    spike_data["x_position_cm"] = x_position_cm
    return spike_data


def add_trial_number(spike_data, raw_position_data):
    trial_number = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_firing_indices = np.asarray(spike_data[spike_data.cluster_id == cluster_id].firing_times)[0]
        trial_number.append(raw_position_data["trial_number"][cluster_firing_indices].to_list())

    spike_data["trial_number"] = trial_number
    return spike_data


def add_trial_type(spike_data, raw_position_data):
    trial_type = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_firing_indices = np.asarray(spike_data[spike_data.cluster_id == cluster_id].firing_times)[0]
        trial_type.append(raw_position_data["trial_type"][cluster_firing_indices].to_list())

    spike_data["trial_type"] = trial_type
    return spike_data


def bin_fr_in_time(spike_data, raw_position_data, smoothen=True):
    if smoothen:
        suffix="_smoothed"
    else:
        suffix=""

    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_time_seconds/settings.time_bin_size)
    n_trials = max(raw_position_data["trial_number"])

    # make an empty list of list for all firing rates binned in time for each cluster
    fr_binned_in_time = [[] for x in range(len(spike_data))]

    # extract spatial variables from raw position
    times = np.array(raw_position_data['time_seconds'], dtype="float64")
    trial_numbers_raw = np.array(raw_position_data['trial_number'], dtype=np.int64)

    # calculate the average fr in each 100ms time bin
    time_bins = np.arange(min(times), max(times), settings.time_bin_size) # 100ms time bins
    tn_time_bin_means = (np.histogram(times, time_bins, weights = trial_numbers_raw)[0] / np.histogram(times, time_bins)[0]).astype(np.int64)

    for i, cluster_id in enumerate(spike_data.cluster_id):
        if len(time_bins)>1:
            spike_times = np.array(spike_data[spike_data["cluster_id"] == cluster_id]["firing_times"].iloc[0])
            spike_times = spike_times/settings.sampling_rate # convert to seconds

            # count the spikes in each time bin and normalise to seconds
            fr_time_bin_means, bin_edges = np.histogram(spike_times, time_bins)
            fr_time_bin_means = fr_time_bin_means/settings.time_bin_size

            # and smooth
            if smoothen:
                fr_time_bin_means = convolve(fr_time_bin_means, gauss_kernel)

            # fill in firing rate array by trial
            fr_binned_in_time_cluster = []
            for trial_number in range(1, n_trials+1):
                fr_binned_in_time_cluster.append(fr_time_bin_means[tn_time_bin_means == trial_number].tolist())
            fr_binned_in_time[i] = fr_binned_in_time_cluster
        else:
            fr_binned_in_time[i] = []

    spike_data["fr_time_binned"+suffix] = fr_binned_in_time
    return spike_data


def bin_fr_in_space(spike_data, raw_position_data, track_length, smoothen=True):
    if smoothen:
        suffix="_smoothed"
    else:
        suffix=""

    vr_bin_size_cm = settings.vr_bin_size_cm
    gauss_kernel = Gaussian1DKernel(settings.guassian_std_for_smoothing_in_space_cm/vr_bin_size_cm)

    # make an empty list of list for all firing rates binned in time for each cluster
    fr_binned_in_space = [[] for x in range(len(spike_data))]
    fr_binned_in_space_bin_centres = [[] for x in range(len(spike_data))]

    elapsed_distance_bins = np.arange(0, (track_length*max(raw_position_data["trial_number"]))+1, vr_bin_size_cm) # might be buggy with anything but 1cm space bins
    trial_numbers_raw = np.array(raw_position_data['trial_number'], dtype=np.int64)
    x_position_elapsed_cm = (track_length*(trial_numbers_raw-1))+np.array(raw_position_data['x_position_cm'], dtype="float64")
    x_dwell_time = np.array(raw_position_data['dwell_time_ms'], dtype="float64")

    for i, cluster_id in enumerate(spike_data.cluster_id):
        if len(elapsed_distance_bins)>1:
            spikes_x_position_cm = np.array(spike_data[spike_data["cluster_id"] == cluster_id]["x_position_cm"].iloc[0])
            trial_numbers = np.array(spike_data[spike_data["cluster_id"] == cluster_id]["trial_number"].iloc[0])

            # convert spike locations into elapsed distance
            spikes_x_position_elapsed_cm = (track_length*(trial_numbers-1))+spikes_x_position_cm

            # count the spikes in each space bin and normalise by the total time spent in that bin for the trial
            fr_hist, bin_edges = np.histogram(spikes_x_position_elapsed_cm, elapsed_distance_bins)
            fr_hist = fr_hist/(np.histogram(x_position_elapsed_cm, elapsed_distance_bins, weights=x_dwell_time)[0])

            # get location bin centres and ascribe them to their trial numbers
            bin_centres = 0.5*(bin_edges[1:]+bin_edges[:-1])
            bin_centres_trial_numbers = (bin_centres//track_length).astype(np.int64)+1

            # nans to zero and smooth
            if smoothen:
                fr_hist = convolve(fr_hist, gauss_kernel)

            # fill in firing rate array by trial
            fr_binned_in_space_cluster = []
            fr_binned_in_space_bin_centres_cluster = []
            for trial_number in range(1, max(raw_position_data["trial_number"]+1)):
                fr_binned_in_space_cluster.append(fr_hist[bin_centres_trial_numbers==trial_number].tolist())
                fr_binned_in_space_bin_centres_cluster.append(bin_centres[bin_centres_trial_numbers==trial_number].tolist())

            fr_binned_in_space[i] = fr_binned_in_space_cluster
            fr_binned_in_space_bin_centres[i] = fr_binned_in_space_bin_centres_cluster
        else:
            fr_binned_in_space[i] = []
            fr_binned_in_space_bin_centres[i] = []

    spike_data["fr_binned_in_space"+suffix] = fr_binned_in_space
    spike_data["fr_binned_in_space_bin_centres"] = fr_binned_in_space_bin_centres

    return spike_data



def add_location_and_task_variables(spike_data, raw_position_data, track_length):
    print('I am extracting firing locations for each cluster...')
    spike_data = add_speed(spike_data, raw_position_data)
    spike_data = add_position_x(spike_data, raw_position_data)
    spike_data = add_trial_number(spike_data, raw_position_data)
    spike_data = add_trial_type(spike_data, raw_position_data)

    spike_data = bin_fr_in_time(spike_data, raw_position_data)
    spike_data = bin_fr_in_space(spike_data, raw_position_data, track_length)
    return spike_data


def split_and_add_trial_number(cluster_index, spike_data_movement, spike_data_stationary, spike_data_trial_number,above_threshold_indices,below_threshold_indices):
    spike_data_movement.trial_number.iloc[cluster_index] = spike_data_trial_number[above_threshold_indices]
    spike_data_stationary.trial_number.iloc[cluster_index] = spike_data_trial_number[below_threshold_indices]
    return spike_data_movement, spike_data_stationary


def split_and_add_x_location_cm(cluster_index, spike_data_movement, spike_data_stationary, spike_data_x_location_cm,above_threshold_indices,below_threshold_indices):
    spike_data_movement.x_position_cm.iloc[cluster_index] = spike_data_x_location_cm[above_threshold_indices]
    spike_data_stationary.x_position_cm.iloc[cluster_index] = spike_data_x_location_cm[below_threshold_indices]
    return spike_data_movement, spike_data_stationary


def split_and_add_trial_type(cluster_index, spike_data_movement, spike_data_stationary, spike_data_trial_type,above_threshold_indices,below_threshold_indices):
    spike_data_movement.trial_type.iloc[cluster_index] = spike_data_trial_type[above_threshold_indices]
    spike_data_stationary.trial_type.iloc[cluster_index] = spike_data_trial_type[below_threshold_indices]
    return spike_data_movement, spike_data_stationary


def split_spatial_firing_by_speed(spike_data, spike_data_movement, spike_data_stationary):
    movement_threshold=settings.movement_threshold # 2.5 cm / second

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_firing_indices = np.asarray(spike_data[spike_data.cluster_id == cluster_id].firing_times)[0]
        above_threshold_indices = np.where(np.array(spike_data.speed_per200ms.iloc[cluster_index]) >= movement_threshold)[0]
        below_threshold_indices = np.where(np.array(spike_data.speed_per200ms.iloc[cluster_index]) < movement_threshold)[0]

        spike_data_movement, spike_data_stationary = split_and_add_trial_number(cluster_index, spike_data_movement, spike_data_stationary, np.array(spike_data.trial_number.iloc[cluster_index]), above_threshold_indices, below_threshold_indices)
        spike_data_movement, spike_data_stationary = split_and_add_x_location_cm(cluster_index,spike_data_movement, spike_data_stationary, np.array(spike_data.x_position_cm.iloc[cluster_index]),above_threshold_indices, below_threshold_indices)
        spike_data_movement, spike_data_stationary = split_and_add_trial_type(cluster_index,   spike_data_movement, spike_data_stationary, np.array(spike_data.trial_type.iloc[cluster_index]),   above_threshold_indices, below_threshold_indices)
    return spike_data_movement, spike_data_stationary


def split_spatial_firing_by_trial_type(spike_data):
    print('I am splitting firing locations by trial type...')
    beaconed_position_cm = []
    beaconed_trial_number = []
    nonbeaconed_position_cm = []
    nonbeaconed_trial_number = []
    probe_position_cm = []
    probe_trial_number = []

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        trials = np.array(cluster_df['trial_number'].tolist())
        locations = np.array(cluster_df['x_position_cm'].tolist())
        trial_type = np.array(cluster_df['trial_type'].tolist())

        beaconed_position_cm.append(locations[trial_type == 0])
        beaconed_trial_number.append(trials[trial_type == 0])
        nonbeaconed_position_cm.append(locations[trial_type == 1])
        nonbeaconed_trial_number.append(trials[trial_type == 1])
        probe_position_cm.append(locations[trial_type == 2])
        probe_trial_number.append(trials[trial_type == 2])

    spike_data["beaconed_position_cm"] = beaconed_position_cm
    spike_data["beaconed_trial_number"] = beaconed_trial_number
    spike_data["nonbeaconed_position_cm"] = nonbeaconed_position_cm
    spike_data["nonbeaconed_trial_number"] = nonbeaconed_trial_number
    spike_data["probe_position_cm"] = probe_position_cm
    spike_data["probe_trial_number"] = probe_trial_number
    return spike_data


def process_spatial_firing(spike_data, raw_position_data, track_length):
    spike_data_movement = spike_data.copy()
    spike_data_stationary = spike_data.copy()

    spike_data = add_location_and_task_variables(spike_data, raw_position_data, track_length)
    spike_data = split_spatial_firing_by_trial_type(spike_data)
    print('-------------------------------------------------------------')
    print('spatial firing processed')
    print('-------------------------------------------------------------')
    return spike_data_movement, spike_data_stationary, spike_data


def process_recordings(vr_recording_path_list):
    vr_recording_path_list.sort()

    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            output_path = recording+'/'+settings.sorterName
            stop_threshold, track_length = get_stop_threshold_and_track_length(recording)
            spike_data = pd.read_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

            raw_position_data, position_data = PostSorting.vr_sync_spatial_data.syncronise_position_data(recording, output_path, track_length)
            spike_data = bin_fr_in_time(spike_data, raw_position_data, smoothen=True)
            spike_data = bin_fr_in_time(spike_data, raw_position_data, smoothen=False)
            spike_data = bin_fr_in_space(spike_data, raw_position_data, track_length, smoothen=True)
            spike_data = bin_fr_in_space(spike_data, raw_position_data, track_length, smoothen=False)
            spike_data.to_pickle(recording+"/MountainSort/DataFrames/spatial_firing.pkl")

            print("successfully processed on "+recording)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)



#  for testing
def main():
    print('-------------------------------------------------------------')
    vr_path_list = []
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort6_july2020/vr") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort8_may2021/vr") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort9_Junji/vr") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort7_october2020/vr") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort5/VirtualReality") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort4/VirtualReality") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality") if f.is_dir()])
    #vr_path_list.extend([f.path for f in os.scandir("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort2/VirtualReality") if f.is_dir()])
    vr_path_list = ["/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality/M1_D31_2018-11-01_12-28-25",
                    "/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/VirtualReality/M1_D18_2018-10-13_12-13-31",
                    "/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort5/VirtualReality/M1_D5_2019-06-21_13-33-50",
                    "/mnt/datastore/Harry/Cohort7_october2020/vr/M7_D12_2020-11-13_16-20-54"]
    vr_path_list = ["/mnt/datastore/Harry/Cohort7_october2020/vr/M7_D12_2020-11-13_16-20-54"]
    process_recordings(vr_path_list)

    print("spatial_firing dataframes have been remade")

if __name__ == '__main__':
    main()