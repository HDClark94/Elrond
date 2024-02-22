from Helpers import open_ephys_IO
import os
import numpy as np
import pandas as pd
from scipy import stats
import PostSorting.parameters
import PostSorting.load_snippet_data_opto
import PostSorting.open_field_make_plots
import time
import PostSorting.SALT
import PostSorting.analyse_opto_inhibition


def load_opto_data(recording_to_process, opto_channel):
    is_found, opto_data = False, None
    print('loading opto channel...')
    file_path = recording_to_process + '/' + opto_channel
    if os.path.exists(file_path):
        opto_data = open_ephys_IO.get_data_continuous(file_path)
        is_found = True
    else:
        print('Opto data was not found.')

    return opto_data, is_found


def get_ons_and_offs(opto_data):
    mode = stats.mode(opto_data[::30000])[0][0]
    opto_on, opto_off = np.where(opto_data > 0.2 + mode), np.where(opto_data <= 0.2 + mode)

    return opto_on, opto_off


# check for opto pulses, find ons/offs and return start/end indices
def process_opto_data(recording_to_process, opto_channel):
    opto_on = opto_off = None
    first_pulse_index, last_pulse_index = None, None
    opto_data, is_found = load_opto_data(recording_to_process, opto_channel)

    if is_found:
        opto_on, opto_off = get_ons_and_offs(opto_data)
        if not np.asarray(opto_on).size:  # if empty
            is_found = False
        elif np.asarray(opto_on).size < 4500:  # if less than 50 (ie pulses from knocking the Arduino)
            is_found = False
        else:  # find starts/ends of opto pulses
            first_pulse_index, last_pulse_index = min(opto_on[0]), max(opto_on[0])

    return opto_on, opto_off, is_found, first_pulse_index, last_pulse_index



def make_opto_data_frame(opto_on: tuple) -> pd.DataFrame:
    opto_data_frame = pd.DataFrame()
    opto_end_times = np.take(opto_on, np.where(np.diff(opto_on)[0] > 1))
    opto_start_times_from_second = np.take(opto_on, np.where(np.diff(opto_on)[0] > 1)[0] + 1)
    opto_start_times = np.append(opto_on[0][0], opto_start_times_from_second)
    opto_end_times = np.append(opto_end_times, opto_on[0][-1])
    opto_data_frame['opto_start_times'] = opto_start_times
    opto_data_frame['opto_end_times'] = opto_end_times

    return opto_data_frame


# find width of opto pulses based on first 50 pulses
def find_pulse_width(starts, ends, fs):
    stimulation_frequency, width = None, None
    if len(starts) > 50:
        widths, betweens = [], []
        for i in range(1, 51):  # calculates width and time between pulses for first 50 pulses (exc. first pulse)
            widths.append(int(((ends[i] - starts[i]) / fs) * 1000))  # pulse widths
            betweens.append(int(((starts[i + 1] - ends[i]) / fs) * 1000))  # time between pulses
        width, between = stats.mode(widths)[0][0], stats.mode(betweens)[0][0]  # mode of each array
        stimulation_frequency = round(1000 / (width + between), 1)  # round to first decimal

    return width, stimulation_frequency


# calculate size of window for plotting/analysis - default is 200 ms
def find_window_size(stimulation_frequency):
    window_size = 200
    if stimulation_frequency is None:
        pass
    elif stimulation_frequency > 5:  # 200 ms window is appropriate for < 5 Hz stimulations
        window_size = 1000 / stimulation_frequency  # calculate window size
        if window_size % 2 != 0:  # check for parity of window
            print("Window size calculated for opto analysis was not divisible by 2...")
            stimulation_frequency, window_size = None, 200  # return to None and use default window

    return stimulation_frequency, int(window_size)


# calculate stimulation frequency from time between pulses, return pulse width (ms) and frequency (Hz)
def find_stimulation_frequency(opto_on, sampling_rate):
    end_times = np.take(opto_on, np.where(np.diff(opto_on)[0] > 1)[0])
    start_times_from_second = np.take(opto_on, np.where(np.diff(opto_on)[0] > 1)[0] + 1)
    start_times = np.append(opto_on[0][0], start_times_from_second)
    pulse_width_ms, stimulation_frequency = find_pulse_width(start_times, end_times, sampling_rate)
    stimulation_frequency, window_size_ms = find_window_size(stimulation_frequency)

    if stimulation_frequency:
        print('Stimulation frequency is', stimulation_frequency, 'Hz, where each pulse is', pulse_width_ms, 'ms wide')
        print('I will use a window of', window_size_ms, 'ms for plotting.')

    else:
        print('Stimulation frequency cannot be determined. Default window size of 200 ms will be used for plotting.')

    return window_size_ms


def load_parameters(prm):
    output_path = prm.get_output_path()
    sampling_rate = prm.get_sampling_rate()
    local_recording_folder = prm.get_local_recording_folder_path()
    sorter_name = prm.get_sorter_name()
    stitchpoint = prm.stitchpoint
    paired_order = prm.paired_order
    dead_channels = prm.get_dead_channels()

    return output_path, sampling_rate, local_recording_folder, sorter_name, stitchpoint, paired_order, dead_channels


def save_opto_metadata(opto_params_is_found, opto_parameters, output_path, window_size_ms, first_spike_latency_ms):
    save_path = '/DataFrames/opto_parameters.pkl'

    if opto_params_is_found:
        opto_parameters['window_size_ms'] = window_size_ms
        opto_parameters['first_spike_latency_ms'] = first_spike_latency_ms
        opto_parameters.to_pickle(output_path + save_path)
    
   
def get_opto_parameters(path_to_recording, output_path, window_size, first_spike_latency):
    found = False
    opto_parameters = np.nan

    for file_name in os.listdir(path_to_recording):
        if file_name == 'opto_parameters.csv':
            print('I found the opto parameters file.')
            found = True
            opto_parameters_path = path_to_recording + file_name
            opto_parameters = pd.read_csv(opto_parameters_path)

    if not found:
        print('There is no opto parameters file, I will assume they are all the same intensity.')

    save_opto_metadata(found, opto_parameters, output_path, window_size, first_spike_latency)

    return opto_parameters, found


# read in opto data from .pkl file
def get_peristimulus_opto_data(window_size_ms, output_path, sampling_rate):
    print('I am getting data for peristimulus array.')
    pulses = pd.read_pickle(output_path + '/DataFrames/opto_pulses.pkl')
    on_pulses = pulses.opto_start_times  # start times for pulses
    window_size_sampling_rate = int(sampling_rate / 1000 * window_size_ms)

    return on_pulses, window_size_sampling_rate


def get_firing_times(cell):
    if 'firing_times_opto' in cell:
        firing_times = np.append(cell.firing_times, cell.firing_times_opto)
    else:
        firing_times = cell.firing_times

    return firing_times


def find_spike_positions_in_window(pulse, firing_times, window_size_sampling_rate):
    spikes_in_window_binary = np.zeros(window_size_sampling_rate)
    window_start, window_end = int(pulse - window_size_sampling_rate / 2), int(pulse + window_size_sampling_rate / 2)
    spikes_in_window_indices = np.where((firing_times > window_start) & (firing_times < window_end))
    spike_times = np.take(firing_times, spikes_in_window_indices)[0]
    position_of_spikes = spike_times.astype(int) - window_start
    spikes_in_window_binary[position_of_spikes] = 1

    return spikes_in_window_binary


# create dataframe that contains peristimulus spikes in specified window (default 200 ms)
def make_peristimulus_df(spatial_firing, on_pulses, window_size_sampling_rate, output_path):
    print('I am making the peristimulus data frame...')
    start_time = time.time()
    peristimulus_spikes_path = output_path + '/DataFrames/peristimulus_spikes.pkl'
    columns = np.append(['session_id', 'cluster_id'], range(window_size_sampling_rate))
    peristimulus_spikes = pd.DataFrame(columns=columns)
    peristimulus_spikes.session_id = spatial_firing.session_id.repeat(len(on_pulses))
    peristimulus_spikes.cluster_id = spatial_firing.cluster_id.repeat(len(on_pulses))
    number_of_rows = len(on_pulses) * len(spatial_firing.cluster_id.unique())
    peristimulus_spikes_binary = np.empty([number_of_rows, window_size_sampling_rate])
    row_number_in_binary_array = 0

    for cell_index, cell in spatial_firing.iterrows():
        for pulse in on_pulses:
            firing_times = get_firing_times(cell)
            spikes_in_window_binary = find_spike_positions_in_window(pulse, firing_times, window_size_sampling_rate)
            peristimulus_spikes_binary[row_number_in_binary_array] = spikes_in_window_binary
            row_number_in_binary_array += 1

    peristimulus_spikes.iloc[:, 2:] = peristimulus_spikes_binary
    peristimulus_spikes.to_pickle(peristimulus_spikes_path)
    elapsed_time = time.time() - start_time
    print('making the peristimulus df took:' + str(elapsed_time))

    return peristimulus_spikes


def get_first_spike_and_latency_for_pulse(firing_times, pulse, first_spike_latency):
    """
    :param firing_times: array of firing times corresponding to a cell (sampling points)
    :param pulse: time point when the light turned on
    :param first_spike_latency: latency window to include spikes from (default is 10ms)
    :return: time points corresponding to the first spike after stimulation and their latencies relative to the pulse
    """
    if len(firing_times[firing_times > pulse]) > 0:  # make sure there are spikes after the pulse
        first_spike_after_pulse = firing_times[firing_times > pulse][0]
        latency = first_spike_after_pulse - pulse
        if latency > first_spike_latency:
            latency = np.nan  # the spike is more than 10 ms after the pulse
            first_spike_after_pulse = np.nan
    else:
        first_spike_after_pulse = np.nan
        latency = np.nan

    return first_spike_after_pulse, latency


def assert_firing_times_is_sorted(firing_times):
    """
    Assert that the time points in firing_times are sorted. This can happen if someone is trying to use the function on
    data that was filtered based on spatial properties (such as taking spikes from a firing field).
    :param firing_times: List of time points when the cell fired
    """
    is_sorted = np.all(np.diff(firing_times) >= 0)
    assert is_sorted, ('The firing times array is not sorted. It has to be sorted for this function to work.', firing_times)


def add_first_spike_times_after_stimulation(spatial_firing, on_pulses, first_spike_latency=300):
    # Identifies first spike firing times and latencies and makes columns ('spike_times_after_opto' and 'latencies')
    print('I will find the first spikes after the light for each opto stimulation pulse.')
    first_spikes_times, latencies = [], []

    for cluster_index, cluster in spatial_firing.iterrows():
        firing_times = cluster.firing_times_opto
        assert_firing_times_is_sorted(firing_times)
        first_spikes_times_cell, latencies_cell = [], []

        for pulse in on_pulses:
            first_spike_after_pulse, latency = get_first_spike_and_latency_for_pulse(firing_times, pulse, first_spike_latency)
            first_spikes_times_cell.append(first_spike_after_pulse)
            latencies_cell.append(latency)

        first_spikes_times.append(first_spikes_times_cell)
        latencies.append(latencies_cell)

    spatial_firing['spike_times_after_opto'] = first_spikes_times
    spatial_firing['opto_latencies'] = latencies

    return spatial_firing


def analyse_latencies(spatial_firing, sampling_rate):
    print('Analysing latencies...')
    latencies_mean_ms, latencies_sd_ms = [], []

    for cluster_index, cluster in spatial_firing.iterrows():
        mean = np.nanmean(cluster.opto_latencies) / sampling_rate * 1000
        sd = np.nanstd(cluster.opto_latencies) / sampling_rate * 1000
        latencies_mean_ms.append(mean)
        latencies_sd_ms.append(sd)

    spatial_firing['opto_latencies_mean_ms'] = latencies_mean_ms
    spatial_firing['opto_latencies_sd_ms'] = latencies_sd_ms

    return spatial_firing


# main function for analysing spike latency around light pulses
def process_spikes_around_light(spatial_firing, prm, window_size_ms=200, first_spike_latency_ms=10, subset=False, pulses=None, window_fs=None, segment_id=0):
    output_path, sampling_rate, local_recording_folder, sorter_name, stitchpoint, paired_order, dead_channels = load_parameters(prm)

    if not subset:
        print('I will process spikes around light...')
        path_to_recording = '/'.join(output_path.split('/')[:-1]) + '/'
        get_opto_parameters(path_to_recording, output_path, window_size_ms, first_spike_latency_ms)
        on_pulses, window_size_sampling_rate = get_peristimulus_opto_data(window_size_ms, output_path, sampling_rate)  # read in opto data

    else:
        on_pulses, window_size_sampling_rate = pulses, window_fs

    peristimulus_spikes = make_peristimulus_df(spatial_firing, on_pulses, window_size_sampling_rate, output_path)  # get peristimulus spikes
    first_spike_latency = sampling_rate / 1000 * first_spike_latency_ms  # in sampling points
    spatial_firing = add_first_spike_times_after_stimulation(spatial_firing, on_pulses, first_spike_latency=first_spike_latency)
    spatial_firing = analyse_latencies(spatial_firing, sampling_rate)
    spatial_firing = PostSorting.load_snippet_data_opto.get_opto_snippets(spatial_firing, local_recording_folder, sorter_name, dead_channels, random_snippets=True, segment_id=segment_id)
    spatial_firing = PostSorting.load_snippet_data_opto.get_opto_snippets(spatial_firing, local_recording_folder, sorter_name, dead_channels, random_snippets=True, column_name='first_spike_snippets_opto', firing_times_column='spike_times_after_opto',segment_id=segment_id)
    spatial_firing = PostSorting.SALT.run_salt_test_on_peristimulus_data(spatial_firing, peristimulus_spikes)
    spatial_firing = PostSorting.analyse_opto_inhibition.run_test_for_opto_inhibition(spatial_firing, peristimulus_spikes)

    return spatial_firing

