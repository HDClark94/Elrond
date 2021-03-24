import open_ephys_IO
import os
import numpy as np
import pandas as pd
from scipy import stats
import PostSorting.parameters
import PostSorting.load_snippet_data_opto

import PostSorting.open_field_make_plots
# import PostSorting.SALT


def load_opto_data(recording_to_process, prm):
    is_found = False
    opto_data = None
    print('loading opto channel...')
    file_path = recording_to_process + '/' + prm.get_opto_channel()
    if os.path.exists(file_path):
        opto_data = open_ephys_IO.get_data_continuous(prm, file_path)
        is_found = True
    else:
        print('Opto data was not found.')
    return opto_data, is_found


def get_ons_and_offs(opto_data):
    # opto_on = np.where(opto_data > np.min(opto_data) + 10 * np.std(opto_data))
    # opto_off = np.where(opto_data <= np.min(opto_data) + 10 * np.std(opto_data))
    mode = stats.mode(opto_data[::30000])[0][0]
    opto_on = np.where(opto_data > 0.2 + mode)
    opto_off = np.where(opto_data <= 0.2 + mode)
    return opto_on, opto_off


def process_opto_data(recording_to_process, prm):
    opto_on = opto_off = None
    opto_data, is_found = load_opto_data(recording_to_process, prm)
    if is_found:
        opto_on, opto_off = get_ons_and_offs(opto_data)
        if not np.asarray(opto_on).size:
            prm.set_opto_tagging_start_index(None)
            is_found = None
        else:
            first_opto_pulse_index = min(opto_on[0])
            prm.set_opto_tagging_start_index(first_opto_pulse_index)

    else:
        prm.set_opto_tagging_start_index(None)

    return opto_on, opto_off, is_found


def make_opto_data_frame(opto_on: tuple) -> pd.DataFrame:
    opto_data_frame = pd.DataFrame()
    opto_end_times = np.take(opto_on, np.where(np.diff(opto_on)[0] > 1))
    opto_start_times_from_second = np.take(opto_on, np.where(np.diff(opto_on)[0] > 1)[0] + 1)
    opto_start_times = np.append(opto_on[0][0], opto_start_times_from_second)
    opto_data_frame['opto_start_times'] = opto_start_times
    opto_end_times = np.append(opto_end_times, opto_on[0][-1])
    opto_data_frame['opto_end_times'] = opto_end_times
    return opto_data_frame


def check_parity_of_window_size(window_size_ms):
    if window_size_ms % 2 != 0:
        print("Window size must be divisible by 2")
        assert window_size_ms % 2 == 0


def get_on_pulse_times(output_path):
    path_to_pulses = output_path + '/DataFrames/opto_pulses.pkl'
    pulses = pd.read_pickle(path_to_pulses)
    on_pulses = pulses.opto_start_times
    return on_pulses


def get_firing_times(cell):
    if 'firing_times_opto' in cell:
        firing_times = np.append(cell.firing_times, cell.firing_times_opto)
    else:
        firing_times = cell.firing_times
    return firing_times


def find_spike_positions_in_window(pulse, firing_times, window_size_sampling_rate):
    spikes_in_window_binary = np.zeros(window_size_sampling_rate)
    window_start = int(pulse - window_size_sampling_rate / 2)
    window_end = int(pulse + window_size_sampling_rate / 2)
    spikes_in_window_indices = np.where((firing_times > window_start) & (firing_times < window_end))
    spike_times = np.take(firing_times, spikes_in_window_indices)[0]
    position_of_spikes = spike_times.astype(int) - window_start
    spikes_in_window_binary[position_of_spikes] = 1
    return spikes_in_window_binary


def make_df_to_append_for_pulse(session_id, cluster_id, spikes_in_window_binary, window_size_sampling_rate):
    columns = np.append(['session_id', 'cluster_id'], range(window_size_sampling_rate))
    df_row = np.append([session_id, cluster_id], spikes_in_window_binary.astype(int))
    df_to_append = pd.DataFrame([(df_row)], columns=columns)
    return df_to_append


def get_peristumulus_opto_data(window_size_ms, output_path, sampling_rate):
    print('Get data for peristimulus array.')
    check_parity_of_window_size(window_size_ms)
    on_pulses = get_on_pulse_times(output_path)  # these are the start times of the pulses
    window_size_sampling_rate = int(sampling_rate/1000 * window_size_ms)
    return on_pulses, window_size_sampling_rate


def make_peristimulus_df(spatial_firing, on_pulses, window_size_sampling_rate, output_path):
    print('Make peristimulus data frame.')
    peristimulus_spikes_path = output_path + '/DataFrames/peristimulus_spikes.pkl'
    columns = np.append(['session_id', 'cluster_id'], range(window_size_sampling_rate))
    peristimulus_spikes = pd.DataFrame(columns=columns)

    for index, cell in spatial_firing.iterrows():
        session_id = cell.session_id
        cluster_id = cell.cluster_id
        if len(on_pulses) >= 500:
            on_pulses = on_pulses[-500:]  # only look at last 500 to make sure it's just the opto tagging
        for pulse in on_pulses:
            firing_times = get_firing_times(cell)
            spikes_in_window_binary = find_spike_positions_in_window(pulse, firing_times, window_size_sampling_rate)
            df_to_append = make_df_to_append_for_pulse(session_id, cluster_id, spikes_in_window_binary, window_size_sampling_rate)
            peristimulus_spikes = peristimulus_spikes.append(df_to_append)
    peristimulus_spikes.to_pickle(peristimulus_spikes_path)
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


def check_if_firing_times_is_sorted(firing_times):
    try:
        assert np.all(np.diff(firing_times) >= 0)
    except AssertionError as error:
        error.args += ('The firing times array is not sorted. It has to be sorted for this function to work.', firing_times)
        raise


def add_first_spike_times_after_stimulation(spatial_firing, on_pulses, first_spike_latency=300):
    # Identifies first spike firing times and latencies and makes columns ('spike_times_after_opto' and 'latencies')
    print('I will find the first spikes after the light for each opto stimulation pulse.')
    first_spikes_times = []
    latencies = []
    for cluster_index, cluster in spatial_firing.iterrows():
        firing_times = cluster.firing_times_opto
        check_if_firing_times_is_sorted(firing_times)
        first_spikes_times_cell = []
        latencies_cell = []
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
    print('Analyse latencies.')
    latencies_mean_ms = []
    latencies_sd_ms = []

    for cluster_index, cluster in spatial_firing.iterrows():
        mean = np.nanmean(cluster.opto_latencies) / sampling_rate * 1000
        sd = np.nanstd(cluster.opto_latencies) / sampling_rate * 1000
        latencies_mean_ms.append(mean)
        latencies_sd_ms.append(sd)
    spatial_firing['opto_latencies_mean_ms'] = latencies_mean_ms
    spatial_firing['opto_latencies_sd_ms'] = latencies_sd_ms
    return spatial_firing


def get_opto_parameters(output_path):
    path_to_recording = '/'.join(output_path.split('/')[:-1]) + '/'
    found = False
    opto_parameters = np.nan
    for file_name in os.listdir(path_to_recording):
        if file_name.startswith("opto_para"):   # we have a fair amount of typos in the file names
            print(file_name)
            print('I found the opto parameters file.')
            found = True
            opto_parameters_path = path_to_recording + file_name
            opto_parameters = pd.read_csv(opto_parameters_path)
    if not found:
        print('There is no opto parameters file, I will assume they are all the same intensity and plot them together.')
    return opto_parameters, found


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
    if opto_params_is_found:
        opto_parameters['window_size_ms'] = window_size_ms
        opto_parameters['first_spike_latency_ms'] = first_spike_latency_ms
        opto_parameters.to_pickle(output_path + '/DataFrames/opto_parameters.pkl')


def process_spikes_around_light(spatial_firing, prm, window_size_ms=40, first_spike_latency_ms=10):
    output_path, sampling_rate, local_recording_folder, sorter_name, stitchpoint, paired_order, dead_channels = load_parameters(prm)
    print('I will process opto data.')
    opto_parameters, opto_params_is_found = get_opto_parameters(output_path)
    save_opto_metadata(opto_params_is_found, opto_parameters, output_path, window_size_ms, first_spike_latency_ms)
    on_pulses, window_size_sampling_rate = get_peristumulus_opto_data(window_size_ms, output_path, sampling_rate)
    first_spike_latency_sampling_points = sampling_rate / 1000 * first_spike_latency_ms
    spatial_firing = add_first_spike_times_after_stimulation(spatial_firing, on_pulses, first_spike_latency=first_spike_latency_sampling_points)
    spatial_firing = analyse_latencies(spatial_firing, sampling_rate)
    spatial_firing = PostSorting.load_snippet_data_opto.get_opto_snippets(spatial_firing, local_recording_folder, sorter_name, stitchpoint, paired_order, dead_channels, random_snippets=True)
    spatial_firing = PostSorting.load_snippet_data_opto.get_opto_snippets(spatial_firing, local_recording_folder, sorter_name, stitchpoint, paired_order, dead_channels, random_snippets=True, column_name='first_spike_snippets_opto', firing_times_column='spike_times_after_opto')
    peristimulus_spikes = make_peristimulus_df(spatial_firing, on_pulses, window_size_sampling_rate, output_path)
    # plt.plot((peristimulus_spikes.iloc[:, 2:].astype(int)).sum().rolling(50).sum())
    # baseline, test = create_baseline_and_test_epochs(peristimulus_spikes)
    # latencies, p_values, I_values = salt(baseline_trials, test_trials, winsize=0.01 * pq.s, latency_step=0.01 * pq.s)

    return spatial_firing



