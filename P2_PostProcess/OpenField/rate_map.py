import numpy as np
import pandas as pd
import time
import settings
from P2_PostProcess.OpenField.spatial_data import get_bin_size, get_dwell, get_number_of_bins
from Helpers.math_utility import gaussian_kernel

def rate_map(cluster_id, smooth, spike_data, positions_x, positions_y,
             number_of_bins_x, number_of_bins_y, bin_size_pixels,
             min_dwell, min_dwell_distance_pixels, dt_position_ms):
    print('calculating the rate map for cluster', str(cluster_id))

    cluster_firing_data_spatial = spike_data[spike_data.cluster_id == cluster_id]
    cluster_firings = pd.DataFrame({'position_x': cluster_firing_data_spatial.position_x_pixels.iloc[0],
                                    'position_y': cluster_firing_data_spatial.position_y_pixels.iloc[0]})

    spike_positions_x = cluster_firings.position_x.values
    spike_positions_y = cluster_firings.position_y.values

    firing_rate_map = np.zeros((number_of_bins_x, number_of_bins_y))
    occupancy_map = np.zeros((number_of_bins_x, number_of_bins_y))
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_pixels + (bin_size_pixels / 2)
            py = y * bin_size_pixels + (bin_size_pixels / 2)
            spike_distances = np.sqrt(np.power(px - spike_positions_x, 2)
                                    + np.power(py - spike_positions_y, 2))
            spike_distances = spike_distances[~np.isnan(spike_distances)]
            occupancy_distances = np.sqrt(np.power((px - positions_x), 2)
                                        + np.power((py - positions_y), 2))
            occupancy_distances = occupancy_distances[~np.isnan(occupancy_distances)]
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_pixels)[0])

            if bin_occupancy >= min_dwell:
                occupancy_map[x,y] = 1
                firing_rate_map[x, y] = sum(gaussian_kernel(spike_distances/smooth)) / \
                                        (sum(gaussian_kernel(occupancy_distances/smooth)) * (dt_position_ms/1000))
            else:
                occupancy_map[x,y] = 0
                firing_rate_map[x, y] = 0

    return firing_rate_map, occupancy_map


def rate_map_vectorised(cluster_id, smooth, spike_data, positions_x, positions_y,
                        number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell,
                        min_dwell_distance_pixels, dt_position_ms):
    print('calculating the rate map for cluster', str(cluster_id))

    cluster_firing_data_spatial = spike_data[spike_data.cluster_id == cluster_id]
    cluster_firings = pd.DataFrame({'position_x': cluster_firing_data_spatial.position_x_pixels.iloc[0],
                                    'position_y': cluster_firing_data_spatial.position_y_pixels.iloc[0]})

    spike_positions_x = cluster_firings.position_x.values
    spike_positions_y = cluster_firings.position_y.values

    spike_positions_y = spike_positions_y[~np.isnan(spike_positions_y)]
    spike_positions_x = spike_positions_x[~np.isnan(spike_positions_x)]
    positions_y = positions_y[~np.isnan(positions_y)]
    positions_x = positions_x[~np.isnan(positions_x)]

    x = np.linspace((bin_size_pixels/2), (bin_size_pixels*number_of_bins_x)-(bin_size_pixels/2), number_of_bins_x)
    y = np.linspace((bin_size_pixels/2), (bin_size_pixels*number_of_bins_y)-(bin_size_pixels/2), number_of_bins_y)

    xv, yv = np.meshgrid(x, y)

    xv_spikes = np.repeat(xv[:, :, np.newaxis], len(spike_positions_x), axis=2)
    yv_spikes = np.repeat(yv[:, :, np.newaxis], len(spike_positions_y), axis=2)
    xv_spikes = xv_spikes - spike_positions_x
    yv_spikes = yv_spikes - spike_positions_y
    xv_spikes = np.power(xv_spikes, 2)
    yv_spikes = np.power(yv_spikes, 2)

    xy_spikes = xv_spikes+yv_spikes
    xy_spikes = np.sqrt(xy_spikes)
    xy_spikes = xy_spikes/smooth
    xy_spikes = gaussian_kernel(xy_spikes)
    xy_spikes = np.sum(xy_spikes, axis=2)
    xy_spikes[np.isnan(xy_spikes)] = 0

    xv_locs = np.repeat(xv[:, :, np.newaxis], len(positions_x), axis=2)
    yv_locs = np.repeat(yv[:, :, np.newaxis], len(positions_y), axis=2)
    xv_locs = xv_locs - positions_x
    yv_locs = yv_locs - positions_y
    xv_locs = np.power(xv_locs, 2)
    yv_locs = np.power(yv_locs, 2)

    xy_locs = xv_locs+yv_locs
    xy_locs = np.sqrt(xy_locs)

    occupancies = np.sum((xy_locs<min_dwell_distance_pixels).astype(int), axis=2)
    occupancies[occupancies < min_dwell] = 0
    occupancies[occupancies!=0] = 1

    xy_locs = xy_locs/smooth
    xy_locs = gaussian_kernel(xy_locs)
    xy_locs = np.sum(xy_locs, axis=2)
    xy_locs[np.isnan(xy_locs)] = 0
    xy_locs = xy_locs*(dt_position_ms/1000)

    firing_rate_map = np.divide(xy_spikes, xy_locs)
    firing_rate_map = firing_rate_map*occupancies # occupancies is a mask

    return np.transpose(firing_rate_map), np.transpose(occupancies)


def calculate_rate_maps(spike_data, spatial_data):
    print('I will calculate firing rate maps now.')
    dt_position_ms = spatial_data.synced_time.diff().mean()*1000
    min_dwell, min_dwell_distance_pixels = get_dwell(spatial_data)
    smooth = 5 / 100 * settings.pixel_ratio
    bin_size_pixels = get_bin_size()
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data)

    firing_rate_maps = []
    max_firing_rates = []
    occupancy_maps = []
    for i, cluster_id in enumerate(spike_data["cluster_id"]):
        if settings.use_vectorised_rate_map_function:
            firing_rate_map, occupancy_map = rate_map_vectorised(cluster_id, smooth, spike_data, spatial_data.position_x_pixels.values,
                                                                 spatial_data.position_y_pixels.values, number_of_bins_x, number_of_bins_y, bin_size_pixels,
                                                                 min_dwell, min_dwell_distance_pixels, dt_position_ms)
        else:
            firing_rate_map, occupancy_map = rate_map(cluster_id, smooth, spike_data, spatial_data.position_x_pixels.values,
                                                      spatial_data.position_y_pixels.values, number_of_bins_x, number_of_bins_y, bin_size_pixels,
                                                      min_dwell, min_dwell_distance_pixels, dt_position_ms)
        firing_rate_maps.append(firing_rate_map)
        occupancy_maps.append(occupancy_map)
        max_firing_rates.append(np.nanmax(firing_rate_map.flatten()))

    spike_data['firing_maps'] = firing_rate_maps
    spike_data['max_firing_rate'] = max_firing_rates
    spike_data['occupancy_maps'] = occupancy_maps
    return spike_data


def test_compare_rate_map_functions():
    n_spikes = 100000
    positions_x = np.random.uniform(0, 400, n_spikes)
    positions_y = np.random.uniform(0, 400, n_spikes)
    firing_data_spatial = pd.DataFrame()
    firing_data_spatial['cluster_id'] = pd.Series(np.array([1]))
    firing_data_spatial['position_x_pixels'] = [np.random.uniform(0, 400, n_spikes)]
    firing_data_spatial['position_y_pixels'] = [np.random.uniform(0, 400, n_spikes)]

    smooth = 22.0
    number_of_bins_x = 42
    number_of_bins_y = 42
    bin_size_pixels = 11
    min_dwell = 3.0
    min_dwell_distance_pixels = 22
    dt_position_ms = 33
    cluster_id = 1

    time_0 = time.time()
    firing_rate_map_old, _ = rate_map(cluster_id, smooth,
                                      firing_data_spatial,
                                      positions_x, positions_y,
                                      number_of_bins_x, number_of_bins_y,
                                      bin_size_pixels, min_dwell,
                                      min_dwell_distance_pixels,
                                      dt_position_ms)

    print("Non vectorized rate map took ", str(time.time()-time_0), " seconds")
    time_0 = time.time()

    firing_rate_map_new, _ = rate_map_vectorised(cluster_id, smooth,
                                                 firing_data_spatial,
                                                 positions_x, positions_y,
                                                 number_of_bins_x, number_of_bins_y,
                                                 bin_size_pixels, min_dwell,
                                                 min_dwell_distance_pixels,
                                                 dt_position_ms)

    print("Vectorized rate map took ", str(time.time()-time_0), " seconds")

    assert np.allclose(firing_rate_map_old, firing_rate_map_new, rtol=1e-05, atol=1e-08)
    return