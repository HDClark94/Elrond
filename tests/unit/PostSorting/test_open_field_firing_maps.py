import numpy as np
from PostSorting.open_field_firing_maps import *
import pandas as pd
import matplotlib.pyplot as plt
import time

def test_calculate_firing_rate_for_cluster_parallel():
    #firing_data_spatial = pd.read_pickle("/mnt/datastore/Harry/MouseOF/test_recordings/M5_2018-03-06_15-34-44_of/MountainSort/DataFrames/spatial_firing.pkl")
    #spatial = pd.read_pickle("/mnt/datastore/Harry/MouseOF/test_recordings/M5_2018-03-06_15-34-44_of/MountainSort/DataFrames/position.pkl")
    #positions_x = spatial.position_x_pixels.values
    #positions_y = spatial.position_y_pixels.values

    # or dummy data
    n_spikes = 100000
    positions_x = np.random.uniform(0, 400, n_spikes)
    positions_y = np.random.uniform(0, 400, n_spikes)
    firing_data_spatial = pd.DataFrame()
    firing_data_spatial['cluster_id'] = pd.Series(np.array([1]))
    firing_data_spatial['position_x_pixels'] = [np.random.uniform(0, 400, n_spikes)]
    firing_data_spatial['position_y_pixels'] = [np.random.uniform(0, 400, n_spikes)]

    cluster = 0
    smooth = 22.0
    number_of_bins_x = 42
    number_of_bins_y = 42
    bin_size_pixels = 11
    min_dwell = 3.0
    min_dwell_distance_pixels = 22
    dt_position_ms = 33
    cluster_id = 1

    time_0 = time.time()
    firing_rate_map_old, _ = calculate_firing_rate_for_cluster_parallel_old(cluster_id, smooth,
                                                                     firing_data_spatial,
                                                                     positions_x, positions_y,
                                                                     number_of_bins_x, number_of_bins_y,
                                                                     bin_size_pixels, min_dwell,
                                                                     min_dwell_distance_pixels,
                                                                     dt_position_ms)

    print("Non vectorized rate map took ", str(time.time()-time_0), " seconds")
    time_0 = time.time()

    firing_rate_map_new, _ = calculate_firing_rate_for_cluster_parallel(cluster_id, smooth,
                                                                     firing_data_spatial,
                                                                     positions_x, positions_y,
                                                                     number_of_bins_x, number_of_bins_y,
                                                                     bin_size_pixels, min_dwell,
                                                                     min_dwell_distance_pixels,
                                                                     dt_position_ms)

    print("Vectorized rate map took ", str(time.time()-time_0), " seconds")

    assert np.allclose(firing_rate_map_old, firing_rate_map_new, rtol=1e-05, atol=1e-08)

def main():
    test_calculate_firing_rate_for_cluster_parallel()

if __name__ == '__main__':
    main()