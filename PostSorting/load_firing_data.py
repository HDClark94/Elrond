import mdaio
import numpy as np
import os
from pathlib import Path
import pandas as pd
import PreClustering.dead_channels
import data_frame_utility
import settings

def get_firing_info(file_path, sorter_name):
    firing_times_path = file_path + '/Electrophysiology/' + sorter_name + '/firings.mda' # sorter name shouldn't contain path slash
    units_list = None
    firing_info = None
    if os.path.exists(firing_times_path):
        firing_info = mdaio.readmda(firing_times_path)
        units_list = np.unique(firing_info[2])
    else:
        print('I could not find the MountainSort output [firing.mda] file. I will check if the data was sorted earlier.')
        spatial_firing_path = file_path + '/MountainSort/DataFrames/spatial_firing.pkl'
        if os.path.exists(spatial_firing_path):
            spatial_firing = pd.read_pickle(spatial_firing_path)
            os.mknod(file_path + '/sorted_data_exists.txt')
            return units_list, firing_info, spatial_firing
        else:
            print('There are no sorting results available for this recording.')
    return units_list, firing_info, False


# if the recording has dead channels, detected channels need to be shifted to get read channel ids
def correct_detected_ch_for_dead_channels(dead_channels, primary_channels):
    for dead_channel in dead_channels:
        indices_to_add_to = np.where(primary_channels >= dead_channel)
        primary_channels[indices_to_add_to] += 1
    return primary_channels


def correct_for_dead_channels(primary_channels, dead_channels):
    if len(dead_channels) != 0:
        dead_channels = list(map(int, dead_channels[0]))
        primary_channels = correct_detected_ch_for_dead_channels(dead_channels, primary_channels)
    return primary_channels


def get_firing_times_for_recording(firing_times, paired_order, stitchpoint):
    """
    (1) Checks 'paired_order' to determine whether this is data from a single recording or from multiple recordings that
    were stitched together before sorting in order to sort them together. For single recordings, paired order is set to
    None. For combined recordings, paired_order indicates the order of the recordings, so for example if paired_order is
    3, this means that this is the third recording in a series of n.

    (2) Selects firing times from the combined sorting output that belong to the recording currently analysed.
    'firing_times' is the output of the spike sorter and contains all firing events from all recordings sorted together.
    The function uses the 'stitchpoint' - a list of time points where the recordings were stitched together to determine
    which firing events belong to the recording currently analysed and return the times that belong to the recording.
    """
    if paired_order is not None:
        if paired_order == 1:
            firing_times = firing_times[firing_times < stitchpoint[0]]
        else:
            bigger_than_previous_stitch = stitchpoint[paired_order - 2] < firing_times
            smaller_than_next = firing_times < stitchpoint[paired_order - 1]
            firing_times = firing_times[bigger_than_previous_stitch & smaller_than_next]
    return firing_times


def get_stitched_opto_tagging_index(opto_tagging_start_index, paired_order, stitchpoint):
    if paired_order is not None:
        if paired_order > 1:
            time_point_to_add = stitchpoint[paired_order - 2]
            opto_tagging_start_index += time_point_to_add
    return opto_tagging_start_index


def process_firing_times(recording_to_process, sorter_name, dead_channels, paired_order=None, stitchpoint=None, opto_tagging_start_index=None, number_of_channels_neighborhood=4):
    session_id = recording_to_process.split('/')[-1]
    units_list, firing_info, spatial_firing = get_firing_info(recording_to_process, sorter_name)
    if isinstance(spatial_firing, pd.DataFrame):
        firing_data = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'primary_channel', 'firing_times', 'firing_times_opto', 'isolation', 'noise_overlap', 'peak_snr', 'mean_firing_rate', 'random_snippets', 'position_x', 'position_y', 'hd', 'position_x_pixels', 'position_y_pixels', 'speed']].copy()
        return firing_data
    cluster_ids = firing_info[2]
    firing_times = firing_info[1]
    primary_channel = firing_info[0]
    primary_channel = correct_for_dead_channels(primary_channel, dead_channels)
    firing_data = data_frame_utility.df_empty(['session_id', 'cluster_id', 'tetrode', 'primary_channel', 'firing_times', 'firing_times_opto'], dtypes=[str, np.uint8, np.uint8, np.uint8, np.uint64, np.uint64])
    for cluster in units_list:
        cluster_firings_all = firing_times[cluster_ids == cluster]
        cluster_firings_all = get_firing_times_for_recording(cluster_firings_all, paired_order, stitchpoint)
        if opto_tagging_start_index is not None:
            cluster_firings = np.take(cluster_firings_all, np.where(cluster_firings_all < opto_tagging_start_index)[0])
            cluster_firings_opto = np.take(cluster_firings_all, np.where(cluster_firings_all >= opto_tagging_start_index)[0])
        else:
            cluster_firings_opto = []
            cluster_firings = cluster_firings_all
        channel_detected = primary_channel[cluster_ids == cluster][0]
        tetrode = int((channel_detected-1)/number_of_channels_neighborhood + 1)
        ch = int((channel_detected - 1) % number_of_channels_neighborhood + 1)
        firing_data = firing_data.append({
            "session_id": session_id,
            "cluster_id":  int(cluster),
            "tetrode": tetrode,
            "primary_channel": ch,
            "firing_times": cluster_firings,
            "firing_times_opto": cluster_firings_opto
        }, ignore_index=True)
    return firing_data


def create_firing_data_frame(recording_to_process, sorter_name, dead_channels, paired_order=None, stitchpoint=None, opto_tagging_start_index=None):
    spike_data = None
    spike_data = process_firing_times(recording_to_process, sorter_name, dead_channels, paired_order, stitchpoint, opto_tagging_start_index)
    return spike_data

def available_ephys_channels(recording_to_process, prm):
    '''
    :param recording_to_process: absolute path of recroding to sort
    :param prm: PostSorting parameter class
    :return: list of named channels for ephys aquisition
    '''

    shared_ephys_channel_marker = prm.get_shared_ephys_channel_marker()
    all_files_names = [f for f in os.listdir(recording_to_process) if os.path.isfile(os.path.join(recording_to_process, f))]
    all_ephys_file_names = [s for s in all_files_names if shared_ephys_channel_marker in s]

    return all_ephys_file_names


