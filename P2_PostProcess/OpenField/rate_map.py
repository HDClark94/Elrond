import numpy as np
import math
import settings

def make_firing_field_maps(spatial_data, firing_data_spatial):
    position_heat_map = get_position_heatmap(spatial_data)
    firing_data_spatial = get_spike_heatmap_parallel(spatial_data, firing_data_spatial)
    firing_data_spatial = find_maximum_firing_rate(firing_data_spatial)
    return position_heat_map, firing_data_spatial


def calculate_rate_map(spike_data, spatial_data):
    return


def get_dwell(spatial_data):
    min_dwell_distance_cm = 5  # from point to determine min dwell time
    min_dwell_distance_pixels = min_dwell_distance_cm/100 * settings.pixel_ratio
    dt_position_ms = spatial_data.synced_time.diff().mean() * 1000  # average sampling interval in position data (ms)
    min_dwell_time_ms = 3 * dt_position_ms  # this is about 100 ms
    min_dwell = round(min_dwell_time_ms / dt_position_ms)
    return min_dwell, min_dwell_distance_pixels


def get_bin_size():
    bin_size_cm = settings.open_field_bin_size_cm
    bin_size_pixels = bin_size_cm/100*settings.pixel_ratio
    return bin_size_pixels


def get_number_of_bins(spatial_data):
    bin_size_pixels = get_bin_size()
    length_of_arena_x = spatial_data.position_x_pixels[~np.isnan(spatial_data.position_x_pixels)].max()
    length_of_arena_y = spatial_data.position_y_pixels[~np.isnan(spatial_data.position_y_pixels)].max()
    number_of_bins_x = math.ceil(length_of_arena_x / bin_size_pixels)
    number_of_bins_y = math.ceil(length_of_arena_y / bin_size_pixels)
    return number_of_bins_x, number_of_bins_y


def get_position_heatmap(spatial_data):
    min_dwell, min_dwell_distance_pixels = get_dwell(spatial_data)
    bin_size_pixels = get_bin_size()
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data)

    position_heat_map = np.zeros((number_of_bins_x, number_of_bins_y))

    # find value for each bin for heatmap
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_pixels + (bin_size_pixels / 2)
            py = y * bin_size_pixels + (bin_size_pixels / 2)

            occupancy_distances = np.sqrt(np.power((px - spatial_data.position_x_pixels.values), 2) + np.power(
                (py - spatial_data.position_y_pixels.values), 2))
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_pixels)[0])

            if bin_occupancy >= min_dwell:
                position_heat_map[x, y] = bin_occupancy
            else:
                position_heat_map[x, y] = np.nan
    return position_heat_map
