import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from PostSorting.SALT import convert_peristimulus_data_to_baseline_and_test


def get_salt_p_values(peristimulus_spikes, spatial_firing):
    # returns dataframe with SALT test p-values for clusters detected during stimulation
    salt_df = pd.DataFrame()
    opto_clusters = peristimulus_spikes.cluster_id.unique()
    spatial_firing = spatial_firing.loc[spatial_firing['cluster_id'].isin(opto_clusters)]
    salt_ps = []
    for index, cell in spatial_firing.iterrows():
        salt_ps.append(cell.SALT_p[0])
    salt_df['cluster_id'] = opto_clusters
    salt_df['SALT_p'] = salt_ps
    return salt_df


def calculate_window_size_in_ms(peristimulus_data, sampling_rate):
    # calculate size of whole window
    sampling_points_in_window = peristimulus_data.shape[1] - 2
    sampling_points_per_ms = sampling_rate/1000
    window_ms = sampling_points_in_window/sampling_points_per_ms
    return window_ms


def calculate_duration_of_activation(peristimulus_data, num_bins):
    baseline, test = convert_peristimulus_data_to_baseline_and_test(peristimulus_data)
    hist_baseline, bins_bl = np.histogram(np.concatenate(baseline), bins=num_bins)
    hist_test, bins_t = np.histogram(np.concatenate(test), bins=num_bins)
    increased_activity_threshold = np.mean(hist_baseline) + 2 * np.std(hist_baseline)   # calculate baseline mean + 2sd
    increased_activity_in_test_window = hist_test > increased_activity_threshold  # check where test is above bl
    start_of_pulse = np.argmax(increased_activity_in_test_window)  # beginning of increased activity (ms)
    duration = 0
    increased_activity = increased_activity_in_test_window[start_of_pulse]
    i = 1
    while increased_activity:
        duration += 1
        increased_activity = increased_activity_in_test_window[start_of_pulse + i]
        if not increased_activity:
            if increased_activity_in_test_window[start_of_pulse + i + 1]:  # check the next one
                increased_activity = True
                i += 1
                duration += 1
        else:
            i += 1

    return duration, start_of_pulse  # returns duration in ms


def find_end_of_activation(peristimulus_data, cluster_id, sampling_rate):
    peristim_cluster = peristimulus_data[peristimulus_data.cluster_id.astype(int) == int(cluster_id)]
    # calculate whole window size so that bins will correspond to 1 ms -- usually 200 ms
    window_size = calculate_window_size_in_ms(peristimulus_data, sampling_rate)
    num_bins = int(window_size/2)  # ms per side of stimulus
    duration, start_of_response = calculate_duration_of_activation(peristim_cluster, num_bins)
    end_of_response = start_of_response + duration

    return end_of_response


def get_number_of_spikes_around_light(peristimulus_spikes, salt_p_values, window_ms=20, sampling_rate=30000,
                                      salt_p_threshold=0.01):
    # get number of spikes before and after light pulse from each light trial
    window = int(sampling_rate/1000) * window_ms  # 30 sampling points per ms
    spikes_before_light = []
    spikes_after_light = []
    cluster_ids = peristimulus_spikes.cluster_id.unique()
    spikes_around_light = pd.DataFrame()

    for cluster_id in cluster_ids:
        spikes = peristimulus_spikes[peristimulus_spikes.cluster_id == cluster_id].iloc[:, 2:].values
        salt_p = salt_p_values[salt_p_values.cluster_id == cluster_id].iloc[0]['SALT_p']
        middle = int(spikes.shape[1] / 2)
        before_light = spikes[:, middle - window:middle]
        spikes_before_light.append(np.sum(before_light, axis=1))

        # for cells with direct activation, start counting after activation window
        if salt_p < salt_p_threshold:
            shifted_middle = find_end_of_activation(peristimulus_spikes, cluster_id, sampling_rate)  # in ms
            shifted_middle = int(sampling_rate/1000) * shifted_middle  # convert to sampling points
            after_light_and_activation = spikes[:, middle + shifted_middle: middle + shifted_middle + window]
            spikes_after_light.append(np.sum(after_light_and_activation, axis=1))
        else:
            after_light = spikes[:, middle:middle + window]
            spikes_after_light.append(np.sum(after_light, axis=1))

    spikes_around_light['cluster_id'] = cluster_ids
    spikes_around_light['spikes_before_light'] = spikes_before_light
    spikes_around_light['spikes_after_light'] = spikes_after_light

    return spikes_around_light


def analyse_inhibition_of_cells(spikes_around_light, spatial_firing):
    # adds U-value and p-value to spatial firing dataframe for neurons with reduced spiking after stimulus
    u_vals = []
    p_vals = []

    for index, cell in spikes_around_light.iterrows():
        reduced_activity = cell.spikes_before_light.sum() > cell.spikes_after_light.sum()
        if reduced_activity:  # only run analysis on cells with lower spiking after stimulus
            u, p = mannwhitneyu(cell.spikes_before_light.tolist(), cell.spikes_after_light.tolist())
            u_vals.append(u)
            p_vals.append(p)
        else:
            u_vals.append(np.nan)  # add NaN for non-inhibited cells
            p_vals.append(np.nan)

    spatial_firing["inhibition_MW_U"] = u_vals
    spatial_firing["inhibition_MW_p"] = p_vals

    return spatial_firing


def run_test_for_opto_inhibition(spatial_firing, peristimulus_data):
    """
    :return: spatial_firing: spatial firing dataframe with cols added for Mann Whitney U result and p-value

    Counts spikes before and after stimulus
    For cells with fewer spikes after stimulus: returns Mann-Whitney U and p-value
    For cells with more spikes after stimulus: returns NaN value
    For cells with direct activation (sig. SALT p-value), spikes are counted from end of activation

    Default window for analysis is 20 ms around stimulus (arg to get_number_of_spikes_around_light)
    Default sig threshold for SALT test is 0.01 (arg to get_number_of_spikes_around_light)
    """

    salt_p_values = get_salt_p_values(peristimulus_data, spatial_firing)
    spikes_around_light = get_number_of_spikes_around_light(peristimulus_data, salt_p_values)
    spatial_firing = analyse_inhibition_of_cells(spikes_around_light, spatial_firing)
    return spatial_firing
