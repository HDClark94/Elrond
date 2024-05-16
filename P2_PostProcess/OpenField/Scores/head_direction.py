from astropy.stats import rayleightest
import os
import math
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import subprocess
import settings


def moving_sum(array, window):
    ret = np.cumsum(array, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    return ret[window:]


def get_rolling_sum(array_in, window):
    if window > (len(array_in) / 3) - 1:
        print('Window for head-direction histogram is too big, HD plot cannot be made.')
    inner_part_result = moving_sum(array_in, window)
    edges = np.append(array_in[-2 * window:], array_in[: 2 * window])
    edges_result = moving_sum(edges, window)
    end = edges_result[window:math.floor(len(edges_result)/2)]
    beginning = edges_result[math.floor(len(edges_result)/2):-window]
    array_out = np.hstack((beginning, inner_part_result, end))
    return array_out


def get_hd_histogram(angles, window_size=23):
    angles = angles[~np.isnan(angles)]
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    binned_hd, _, _ = plt.hist(angles, theta)
    smooth_hd = get_rolling_sum(binned_hd, window=window_size)
    return smooth_hd


# max firing rate at the angle where the firing rate is highest
def get_max_firing_rate(spatial_firing):
    max_firing_rates = []
    preferred_directions = []
    for index, cluster in spatial_firing.iterrows():
        hd_hist = cluster.hd_spike_histogram
        max_firing_rate = np.max(hd_hist.flatten())
        max_firing_rates.append(max_firing_rate)

        preferred_direction = np.where(hd_hist == max_firing_rate)
        preferred_directions.append(preferred_direction[0])

    spatial_firing['max_firing_rate_hd'] = np.array(max_firing_rates) / 1000  # Hz # TODO why is this here?
    spatial_firing['preferred_HD'] = preferred_directions
    return spatial_firing


def get_hd_score_for_cluster(hd_hist):
    angles = np.linspace(-179, 180, 360)
    angles_rad = angles*np.pi/180
    dy = np.sin(angles_rad)
    dx = np.cos(angles_rad)

    totx = sum(dx * hd_hist)/sum(hd_hist)
    toty = sum(dy * hd_hist)/sum(hd_hist)
    r = np.sqrt(totx*totx + toty*toty)
    return r


'''
This test is used to identify a non-uniform distribution, i.e. it is designed for detecting an unimodal deviation from 
uniformity. More precisely, it assumes the following hypotheses: - H0 (null hypothesis): The population is distributed 
uniformly around the circle. - H1 (alternative hypothesis): The population is not distributed uniformly around the 
circle. Small p-values suggest to reject the null hypothesis.

This is an alternative to using the population mean vector as a head-directions score.

https://docs.astropy.org/en/stable/_modules/astropy/stats/circstats.html#rayleightest
'''

def add_head_direction_rayleigh_score(spike_data):
    print('I will do the Rayleigh test to check if head-direction tuning is uniform.')
    rayleigh_ps = []
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
        hd_hist = cluster_df.hd_spike_histogram.iloc[0].copy()
        bins_in_histogram = len(hd_hist)
        values = np.radians(np.arange(0, 360, int(360 / bins_in_histogram)))
        p = rayleightest(values, weights=hd_hist)
        rayleigh_ps.append(p)
    spike_data['rayleigh_score'] = rayleigh_ps
    return spike_data


def calculate_head_direction_score(spatial_firing):
    hd_scores = []
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        hd_hist = cluster_df.hd_spike_histogram.iloc[0].copy()
        r = get_hd_score_for_cluster(hd_hist)
        hd_scores.append(r)
        print("HD score for cluster", str(cluster_id), ":", str(np.round(r, decimals=2)))
    spatial_firing['hd_score'] = np.array(hd_scores)
    return spatial_firing


def calculate_head_direction_scores(spike_data, spatial_data):
    print('I will process head-direction data now.')
    angles_whole_session = (np.array(spatial_data.hd) + 180) * np.pi / 180
    hd_histogram = get_hd_histogram(angles_whole_session)
    hd_histogram /= settings.sampling_rate

    hd_spike_histograms = []
    for index, cluster in spike_data.iterrows():
        angles_spike = (np.array(cluster.hd) + 180) * np.pi / 180
        hd_spike_histogram = get_hd_histogram(angles_spike)
        hd_spike_histogram = hd_spike_histogram / hd_histogram
        hd_spike_histograms.append(hd_spike_histogram)

    spike_data['hd_spike_histogram'] = hd_spike_histograms
    spike_data = get_max_firing_rate(spike_data)
    spike_data = calculate_head_direction_score(spike_data)
    spike_data = add_head_direction_rayleigh_score(spike_data)
    return spike_data

def main():
    pass

if __name__ == '__main__':
    main()


