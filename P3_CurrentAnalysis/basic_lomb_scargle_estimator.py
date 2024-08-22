# This file makes a basic attempt to make a simplified version of the lomb-scargle approximation of periodicity
# as in Clark and Nolan 2024
import pandas as pd
import numpy as np
import time
from scipy import stats
from scipy import signal
from astropy.timeseries import LombScargle
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.nddata import block_reduce

# Periodgram settings
frequency_step = 0.02
frequency = np.arange(0.1, 10+frequency_step, frequency_step) # spatial freqs to test for
window_length_in_laps = 3 # n trials (laps)
power_estimate_step = 5 # cm

def distance_from_integer(frequencies):
    distance_from_zero = np.asarray(frequencies)%1
    distance_from_one = 1-(np.asarray(frequencies)%1)
    tmp = np.vstack((distance_from_zero, distance_from_one))
    return np.min(tmp, axis=0)

def lomb_scargle(spike_data, processed_position_data, track_length):
    n_trials = len(processed_position_data)
    elapsed_distance_bins = np.arange(0, (track_length*n_trials)+1, 1)
    elapsed_distance = 0.5*(elapsed_distance_bins[1:]+elapsed_distance_bins[:-1])/track_length
    sliding_window_size=track_length*window_length_in_laps # cm

    all_powers = []
    all_centre_distances = []
    all_freqs = []
    all_deltas = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_spike_data = spike_data[spike_data["cluster_id"] == cluster_id]
        firing_rates = np.array(cluster_spike_data["fr_binned_in_space_smoothed"].iloc[0])
        bin_centres = np.array(cluster_spike_data["fr_binned_in_space_bin_centres"].iloc[0])
        firing_times_cluster = np.array(cluster_spike_data["firing_times"].iloc[0])

        time_then = time.time()
        if len(firing_times_cluster)>1:
            fr = firing_rates.flatten()

            powers = []
            centre_distances = []
            indices_to_test = np.arange(0, len(fr)-sliding_window_size, 1, dtype=np.int64)[::power_estimate_step]
            for m in indices_to_test:
                ls = LombScargle(elapsed_distance[m:m+sliding_window_size], fr[m:m+sliding_window_size])
                power = ls.power(frequency)
                powers.append(power.tolist())
                centre_distances.append(np.nanmean(elapsed_distance[m:m+sliding_window_size]))
            powers = np.array(powers)
            centre_distances = np.array(centre_distances)
            centre_trials = np.round(np.array(centre_distances)).astype(np.int64)

            powers[np.isnan(powers)] = 0
            max_power_freqs = []
            for i in range(len(powers)):
                a = frequency[np.nanargmax(powers[i])]
                max_power_freqs.append(a)
            max_power_freqs = np.array(max_power_freqs)
            dist_from_spatial_freq_int = distance_from_integer(max_power_freqs)
            all_powers.append(powers)
            all_centre_distances.append(centre_distances)
            all_freqs.append(max_power_freqs)
            all_deltas.append(dist_from_spatial_freq_int)
        else:
            all_powers.append([])
            all_centre_distances.append([])
            all_freqs.append([])
            all_deltas.append([])

        print(f'lomb scargle computation time for cluster {cluster_id} is {time.time()-time_then}, seconds')
 
    spike_data["ls_powers"] = all_powers
    spike_data["ls_centre_distances"] = all_centre_distances
    spike_data["ls_freqs"] = all_freqs
    spike_data["ls_deltas"] = all_deltas
    return spike_data

def main():
    spike_data = pd.read_pickle("/mnt/datastore/Harry/Cohort11_april2024/derivatives/M21/D26/vr/M21_D26_2024-05-28_17-04-41_VR1/processed/kilosort4/spikes.pkl")
    position_data = pd.read_csv("/mnt/datastore/Harry/Cohort11_april2024/derivatives/M21/D26/vr/M21_D26_2024-05-28_17-04-41_VR1/processed/position_data.csv")
    processed_position_data = pd.read_pickle("/mnt/datastore/Harry/Cohort11_april2024/derivatives/M21/D26/vr/M21_D26_2024-05-28_17-04-41_VR1/processed/processed_position_data.pkl")
    spike_data = lomb_scargle(spike_data, processed_position_data)
    print("done")

if __name__ == '__main__':
    main()