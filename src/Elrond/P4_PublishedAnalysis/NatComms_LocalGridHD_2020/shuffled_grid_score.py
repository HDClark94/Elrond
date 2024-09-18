import glob
import math
import pandas as pd
import numpy as np
import Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings
import os
import PostSorting.open_field_firing_maps
import PostSorting.open_field_grid_cells
import PostSorting.parameters
from scipy import stats

prm = PostSorting.parameters.Parameters()
prm.set_pixel_ratio(440)

local_path_mouse = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/shuffled_grid_analysis/all_mice_df.pkl'
local_path_rat = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/shuffled_grid_analysis/all_rats_df.pkl'
path_to_data = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/shuffled_grid_analysis/'
save_output_path = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/shuffled_grid_analysis/'
server_path_mouse = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_server_path_mouse()
server_path_rat = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_server_path_rat()


def load_spatial_firing(output_path, server_path, animal, spike_sorter=''):
    if os.path.exists(output_path):
        spatial_firing = pd.read_pickle(output_path)
        return spatial_firing
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
        position_data_path = recording_folder + spike_sorter + '/DataFrames/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(data_frame_path)
            position_data = pd.read_pickle(position_data_path)
            if 'grid_score' in spatial_firing:
                if animal == 'rat':
                    spatial_firing = spatial_firing[['session_id', 'cell_id', 'cluster_id', 'firing_times',
                                                    'number_of_spikes', 'grid_score']].copy()
                if animal == 'mouse':
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'firing_times',
                                                     'number_of_spikes', 'hd', 'speed', 'mean_firing_rate',
                                                     'hd_spike_histogram', 'grid_score']].copy()

                spatial_firing['trajectory_x'] = [position_data.position_x] * len(spatial_firing)
                spatial_firing['trajectory_y'] = [position_data.position_y] * len(spatial_firing)
                spatial_firing['synced_time'] = [position_data.synced_time] * len(spatial_firing)
                spatial_firing_data = spatial_firing_data.append(spatial_firing)

    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


# generate shuffled data by shifting spikes in time (20s - recording length - 20s) * 200
def shuffle_data(cell, sampling_rate, movement_sampling_rate):
    firing_times = cell.firing_times
    end_of_movement_data = np.floor(len(cell.trajectory_x) / movement_sampling_rate * sampling_rate)
    times_to_shift_by = np.random.uniform(20*sampling_rate, end_of_movement_data, 200) / sampling_rate
    firing_times_reshaped = firing_times.reshape(1, len(firing_times)) / sampling_rate
    shuffled_times = (np.repeat(firing_times_reshaped, 200, axis=0).T + times_to_shift_by) % (end_of_movement_data / sampling_rate)
    return shuffled_times


def get_position_indices(shuffled_times, movement_sampling_rate):
    indices = np.floor(shuffled_times * movement_sampling_rate).astype(int)
    return indices


def get_position_for_shuffled_times(shuffled_times, cell, movement_sampling_rate):
    indices = get_position_indices(shuffled_times, movement_sampling_rate)
    position_x = np.zeros(indices.shape)
    position_y = np.zeros(indices.shape)
    for shuffle in range(200):
        position_x_shuffle = cell.trajectory_x.values[indices[:, shuffle]]
        position_y_shuffle = cell.trajectory_y.values[indices[:, shuffle]]
        position_x[:, shuffle] = position_x_shuffle
        position_y[:, shuffle] = position_y_shuffle
    return position_x, position_y


def get_position_heat_map(cell):
    spatial_data = pd.DataFrame()
    spatial_data['position_x_pixels'] = cell.trajectory_x
    spatial_data['position_y_pixels'] = cell.trajectory_y
    spatial_data['synced_time'] = cell.synced_time
    position_heat_map = PostSorting.open_field_firing_maps.get_position_heatmap(spatial_data)
    return position_heat_map


def get_number_of_bins(spatial_data, prm):
    bin_size_pixels = PostSorting.open_field_firing_maps.get_bin_size()
    length_of_arena_x = spatial_data.trajectory_x[~np.isnan(spatial_data.trajectory_x)].max()
    length_of_arena_y = spatial_data.trajectory_y[~np.isnan(spatial_data.trajectory_y)].max()
    number_of_bins_x = math.ceil(length_of_arena_x / bin_size_pixels)
    number_of_bins_y = math.ceil(length_of_arena_y / bin_size_pixels)
    return number_of_bins_x, number_of_bins_y


def get_rate_maps(cell, spike_x, spike_y, number_of_shuffles=200):
    dt_position_ms = cell.synced_time.diff().mean() * 1000
    min_dwell, min_dwell_distance_pixels = PostSorting.open_field_firing_maps.get_dwell(cell)
    smooth = 5 / 100 * prm.get_pixel_ratio()
    bin_size_pixels = PostSorting.open_field_firing_maps.get_bin_size()
    number_of_bins_x, number_of_bins_y = get_number_of_bins(cell, prm)
    firing_rate_maps = []
    for shuffle in range(number_of_shuffles):
        spike_positions_x = spike_x[:, shuffle]
        spike_positions_y = spike_y[:, shuffle]
        firing_rate_map = np.zeros((number_of_bins_x, number_of_bins_y))
        for x in range(number_of_bins_x):
            for y in range(number_of_bins_y):
                px = x * bin_size_pixels + (bin_size_pixels / 2)
                py = y * bin_size_pixels + (bin_size_pixels / 2)
                spike_distances = np.sqrt(np.power(px - spike_positions_x, 2) + np.power(py - spike_positions_y, 2))
                spike_distances = spike_distances[~np.isnan(spike_distances)]
                occupancy_distances = np.sqrt(np.power((px - spike_x), 2) + np.power((py - spike_y), 2))
                occupancy_distances = occupancy_distances[~np.isnan(occupancy_distances)]
                bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_pixels)[0])

                if bin_occupancy >= min_dwell:
                    firing_rate_map[x, y] = sum(PostSorting.open_field_firing_maps.gaussian_kernel(spike_distances / smooth)) / (
                                sum(PostSorting.open_field_firing_maps.gaussian_kernel(occupancy_distances / smooth)) * (dt_position_ms / 1000))

                else:
                    firing_rate_map[x, y] = 0
        firing_rate_maps.append(firing_rate_map)
        # firing_rate_map = np.rot90(firing_rate_map)
    return firing_rate_maps


def get_grid_scores(firing_rate_maps):
    grid_scores = []
    for shuffle in range(len(firing_rate_maps)):
        firing_rate_map = firing_rate_maps[shuffle]
        rate_map_correlogram = PostSorting.open_field_grid_cells.get_rate_map_autocorrelogram(firing_rate_map)
        field_properties = PostSorting.open_field_grid_cells.find_autocorrelogram_peaks(rate_map_correlogram)
        if len(field_properties) >= 0:  # todo change back to 7
            grid_spacing, field_size, grid_score = PostSorting.open_field_grid_cells.calculate_grid_metrics(rate_map_correlogram, field_properties)
            grid_scores.append(grid_score)
        else:
            print('Not enough fields to calculate grid metrics.')
            grid_scores.append(np.nan)
    return grid_scores


def process_data(animal):
    if animal == 'mouse':
        spatial_firing = load_spatial_firing(local_path_mouse, server_path_mouse, animal, spike_sorter='\MountainSort')
        sampling_rate = 30000
        movement_sampling_rate = 30
    else:
        spatial_firing = load_spatial_firing(local_path_rat, server_path_rat, animal, spike_sorter='')
        sampling_rate = 50000
        movement_sampling_rate = 50

    grid_percentiles = []

    for index, cell in spatial_firing.iterrows():
        if cell.grid_score > 0.2:
            # position_heat_map = get_position_heat_map(cell)
            shuffled_times_seconds = shuffle_data(cell, sampling_rate, movement_sampling_rate)
            position_x, position_y = get_position_for_shuffled_times(shuffled_times_seconds, cell, movement_sampling_rate)
            rate_maps = get_rate_maps(cell, position_x, position_y)
            grid_scores = get_grid_scores(rate_maps)
            print('grid scores ')
            print(grid_scores)
            grid_percentile = stats.percentileofscore(grid_scores[~np.isnan(grid_scores)], cell.grid_score)
            print(cell.cluster_id)
            print('grid score: ')
            print(cell.grid_score)
            print('grid percentile: ')
            print(grid_percentile)
            grid_percentiles.append(grid_percentile)
    spatial_firing['shuffled_grid_percentile'] = grid_percentiles
    return spatial_firing


# calculate grid score for each shuffle
# find 95th percentile of shuffled grid scores
# check if read grid score is >= to decide if it's a grid cell


def main():
    process_data('mouse')
    process_data('rat')


if __name__ == '__main__':
    main()
