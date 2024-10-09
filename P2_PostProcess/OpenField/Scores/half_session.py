import numpy as np
from scipy.stats.stats import pearsonr
from P2_PostProcess.OpenField.rate_map import rate_map_vectorised, rate_map, get_bin_size, get_number_of_bins, get_dwell
from P2_PostProcess.OpenField.spatial_data import get_position_heatmap
import settings as settings
import pandas as pd


def make_trajectory_heat_maps(whole_trajectory, trajectory_1, trajectory_2):
    min_dwell, min_dwell_distance_pixels = get_dwell(whole_trajectory)
    number_of_bins_x, number_of_bins_y = get_number_of_bins(whole_trajectory)
    position_heat_map_first = get_position_heatmap(trajectory_1,
                                                   number_of_bins_x=number_of_bins_x,
                                                   number_of_bins_y=number_of_bins_y,
                                                   min_dwell_distance_pixels=min_dwell_distance_pixels,
                                                   min_dwell=min_dwell)
    position_heat_map_second = get_position_heatmap(trajectory_2,
                                                    number_of_bins_x=number_of_bins_x,
                                                    number_of_bins_y=number_of_bins_y,
                                                    min_dwell_distance_pixels=min_dwell_distance_pixels,
                                                    min_dwell=min_dwell)
    print('Made trajectory heatmaps for both halves.')
    return position_heat_map_first, position_heat_map_second


def make_same_sized_rate_maps(trajectory_1, trajectory_2, cluster_spatial_firing_1, cluster_spatial_firing_2):
    whole_trajectory = trajectory_1._append(trajectory_2)
    cluster_id = cluster_spatial_firing_1.cluster_id.iloc[0]

    number_of_bins_x, number_of_bins_y = get_number_of_bins(whole_trajectory)
    dt_position_ms = whole_trajectory.synced_time.diff().mean() * 1000
    smooth = 5 / 100 * settings.pixel_ratio
    bin_size_pixels = get_bin_size()
    min_dwell, min_dwell_distance_pixels = get_dwell(whole_trajectory)

    if settings.use_vectorised_rate_map_function:
        rate_map_1, _ = rate_map_vectorised(cluster_id, smooth, cluster_spatial_firing_1,trajectory_1.position_x_pixels.values,
                                            trajectory_1.position_y_pixels.values, number_of_bins_x, number_of_bins_y,
                                            bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms)
        rate_map_2, _ = rate_map_vectorised(cluster_id, smooth, cluster_spatial_firing_1,trajectory_1.position_x_pixels.values,
                                            trajectory_1.position_y_pixels.values, number_of_bins_x, number_of_bins_y,
                                            bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms)
    else:
        rate_map_1, _ = rate_map(cluster_id, smooth, cluster_spatial_firing_1,trajectory_1.position_x_pixels.values,
                                 trajectory_1.position_y_pixels.values, number_of_bins_x, number_of_bins_y,
                                 bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms)
        rate_map_2, _ = rate_map(cluster_id, smooth, cluster_spatial_firing_2,trajectory_2.position_x_pixels.values,
                                 trajectory_2.position_y_pixels.values, number_of_bins_x, number_of_bins_y,
                                 bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms)

    position_heatmap_1, position_heatmap_2 = make_trajectory_heat_maps(whole_trajectory, trajectory_1, trajectory_2)
    return rate_map_1, rate_map_2, position_heatmap_1, position_heatmap_2


def correlate_ratemaps(rate_map_first, rate_map_second, position_heatmap_1, position_heatmap_2):
    print('Correlate rate maps.')
    rate_map_first_flat = rate_map_first.flatten()
    rate_map_second_flat = rate_map_second.flatten()
    position_heatmap_1_flat = position_heatmap_1.flatten()
    position_heatmap_2_flat = position_heatmap_2.flatten()

    mask_for_nans_in_first = ~np.isnan(position_heatmap_1_flat)
    mask_for_nans_in_second = ~np.isnan(position_heatmap_2_flat)
    combined_mask = mask_for_nans_in_first & mask_for_nans_in_second

    rate_map_first_filtered = rate_map_first_flat[combined_mask]
    rate_map_second_filtered = rate_map_second_flat[combined_mask]

    pearson_r, p = pearsonr(rate_map_first_filtered, rate_map_second_filtered)
    percentage_of_excluded_bins = (len(rate_map_first_flat) - len(rate_map_first_filtered)) / len(rate_map_first_flat) * 100
    return pearson_r, percentage_of_excluded_bins


def get_half_of_the_data(spike_data, synced_spatial_data, half='first_half'):
    end_of_first_half_seconds = (synced_spatial_data.synced_time.max() + synced_spatial_data.synced_time.min()) / 2
    end_of_first_half_ephys_sampling_points = end_of_first_half_seconds * settings.sampling_rate

    spike_data_half = pd.DataFrame()
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        if half == 'first_half':
            half_mask = cluster_df['firing_times'].iloc[0] < end_of_first_half_ephys_sampling_points
            half_synced_data_indices = synced_spatial_data.synced_time < end_of_first_half_seconds
        elif half == 'second_half':
            half_mask = cluster_df['firing_times'].iloc[0] >= end_of_first_half_ephys_sampling_points
            half_synced_data_indices = synced_spatial_data.synced_time >= end_of_first_half_seconds

        half_spike_data_cluster = cluster_df.copy()
        half_spike_data_cluster['firing_times'] = [cluster_df['firing_times'].iloc[0][half_mask].copy()]
        half_spike_data_cluster['position_x'] = [np.array(cluster_df['position_x'].iloc[0])[half_mask].copy()]
        half_spike_data_cluster['position_y'] = [np.array(cluster_df['position_y'].iloc[0])[half_mask].copy()]
        half_spike_data_cluster['position_x_pixels'] = [np.array(cluster_df['position_x_pixels'].iloc[0])[half_mask].copy()]
        half_spike_data_cluster['position_y_pixels'] = [np.array(cluster_df['position_y_pixels'].iloc[0])[half_mask].copy()]
        half_spike_data_cluster['hd'] = [np.array(cluster_df['hd'].iloc[0])[half_mask].copy()]
        spike_data_half = pd.concat([spike_data_half, half_spike_data_cluster], ignore_index=True)

    synced_spatial_data_half = synced_spatial_data[half_synced_data_indices].copy()
    return spike_data_half, synced_spatial_data_half



def calculate_half_session_stability_scores(spike_data, spatial_data):
    spike_data_first, synced_spatial_data_first = get_half_of_the_data(spike_data, spatial_data, half='first_half')
    spike_data_second, synced_spatial_data_second = get_half_of_the_data(spike_data, spatial_data, half='second_half')

    pearson_rs = []
    percent_excluded_bins_all = []
    for cluster_index, cluster_id in enumerate(spike_data_first.cluster_id):
        cluster_firsthalf = spike_data_first[spike_data_first.cluster_id == cluster_id]
        cluster_secondhalf = spike_data_second[spike_data_second.cluster_id == cluster_id]

        rate_map_first, rate_map_second, position_heatmap_1, position_heatmap_2 = \
            make_same_sized_rate_maps(synced_spatial_data_first, synced_spatial_data_second, cluster_firsthalf, cluster_secondhalf)
        pearson_r, percentage_of_excluded_bins = correlate_ratemaps(rate_map_first, rate_map_second, position_heatmap_1, position_heatmap_2)

        pearson_rs.append(pearson_r)
        percent_excluded_bins_all.append(percentage_of_excluded_bins)

    spike_data['rate_map_correlation_first_vs_second_half'] = pearson_rs
    spike_data['percent_excluded_bins_rate_map_correlation_first_vs_second_half_p'] = percent_excluded_bins_all
    return spike_data


def main():
    print("============================================")
    print("============================================")

if __name__ == '__main__':
    main()
