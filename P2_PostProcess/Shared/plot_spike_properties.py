from Helpers import array_utility, plot_utility
import os
import matplotlib.pylab as plt
import math
import numpy as np
import pandas as pd
import scipy.ndimage
from typing import Tuple
import settings

def plot_spike_histogram(spatial_firing, output_path, sampling_rate = settings.sampling_rate):
    print('I will plot spikes vs time for the whole session excluding opto tagging.')
    save_path = output_path + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):

        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

        if len(cluster_df['firing_times'].iloc[0])>1: # catches cases where there is no spikes found in the VR but is found in the OF
            number_of_bins = int((cluster_df['firing_times'].iloc[0][-1] - cluster_df['firing_times'].iloc[0][0]) / (5 * sampling_rate))
            firings_cluster =cluster_df['firing_times'].iloc[0] / sampling_rate / 60
            spike_hist = plt.figure()
            spike_hist.set_size_inches(5, 5, forward=True)
            ax = spike_hist.add_subplot(1, 1, 1)
            spike_hist, ax = plot_utility.style_plot(ax)
            if number_of_bins > 0:
                hist, bins = np.histogram(firings_cluster, bins=number_of_bins)
                width = bins[1] - bins[0]
                center = (bins[:-1] + bins[1:]) / 2
                plt.bar(center, hist, align='center', width=width, color='black')
            plt.title('Spike histogram \n total spikes = ' + str(cluster_df['number_of_spikes'].iloc[0]) + ', \n mean fr = ' + str(round(cluster_df['mean_firing_rate'].iloc[0], 0)) + ' Hz', y=1.08, fontsize=24)
            plt.xlabel('Time (min)', fontsize=25)
            plt.ylabel('Number of spikes', fontsize=25)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '_spike_histogram.png', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()



def calculate_autocorrelogram_hist(spikes, bin_size, window):
    half_window = int(window/2)
    number_of_bins = int(math.ceil(spikes[-1]*1000))
    train = np.zeros(number_of_bins)
    bins = np.zeros(len(spikes))

    for spike in range(len(spikes)-1):
        bin = math.floor(spikes[spike]*1000)
        train[bin] = train[bin] + 1
        bins[spike] = bin

    counts = np.zeros(window+1)
    counted = 0
    for b in range(len(bins)):
        bin = int(bins[b])
        window_start = int(bin - half_window)
        window_end = int(bin + half_window + 1)
        if (window_start > 0) and (window_end < len(train)):
            counts = counts + train[window_start:window_end]
            counted = counted + sum(train[window_start:window_end]) - train[bin]

    counts[half_window] = 0
    if max(counts) == 0 and counted == 0:
        counted = 1

    corr = counts / counted
    time = np.arange(-half_window, half_window + 1, bin_size)
    return corr, time


def get_10ms_autocorr(firing_times_cluster, sampling_rate):
    corr1, time1 = calculate_autocorrelogram_hist(np.array(firing_times_cluster) / sampling_rate, 1, 20)
    return corr1, time1

def get_250ms_autocorr(firing_times_cluster, sampling_rate):
    corr, time = calculate_autocorrelogram_hist(np.array(firing_times_cluster) / sampling_rate, 1, 500)
    return corr, time

def make_combined_autocorr_plot(time_10, corr_10, time_250, corr_250, spike_data, save_path, cluster_index, cluster_id):
    grid = plt.GridSpec(2, 1, hspace=0.5)
    autocorr_plot = plt.subplot(grid[0, 0])
    plt.suptitle("Autocorrelograms", fontsize=24)
    plt.xlabel('Time lag (ms)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.xlim(-10, 10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks([-10, 0, 10], [-10, 0, 10])
    plt.bar(time_10, corr_10, align='center', width=1, color='black')

    autocorr_plot2 = plt.subplot(grid[1, 0])
    plt.xlim(-250, 250)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Time lag (ms)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.xticks([-250, 0, 250], [-250, 0, 250])
    plt.bar(time_250, corr_250, align='center', width=1, color='black')
    plt.savefig(save_path + '/' + spike_data.session_id.iloc[cluster_index] + '_' + str(cluster_id) + '_autocorrelograms.png',
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_autocorrelograms(spike_data: pd.DataFrame, output_path: str, sampling_rate = settings.sampling_rate) -> None:
    plt.close()
    print('I will plot autocorrelograms for each cluster (10 ms and 250 ms windows).')
    save_path = output_path + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        firing_times_cluster = cluster_df['firing_times'].iloc[0]
        if len(firing_times_cluster)>1: # only calculate autocorr if there are any spikes
            corr_10, time_10 = get_10ms_autocorr(firing_times_cluster, sampling_rate)
            corr_250, time_250 = get_250ms_autocorr(firing_times_cluster, sampling_rate)
            make_combined_autocorr_plot(time_10, corr_10, time_250, corr_250, spike_data, save_path, cluster_index, cluster_id)


def plot_spikes_for_channel(grid, highest_value, lowest_value, spike_data, cluster_id, channel, snippet_column_name):
    cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster
    snippet_plot = plt.subplot(grid[int(channel/2), channel % 2])
    # waveform shapes are multiplied by -1 to be consistent with how they are shown in most papers
    mean = np.mean(cluster_df[snippet_column_name].iloc[0][channel, :, :], 1) * -1
    plt.ylim(lowest_value - 10, highest_value + 30)
    plot_utility.style_plot(snippet_plot)
    snippet_plot.plot(cluster_df[snippet_column_name].iloc[0][channel, :, :] * -1, color='lightslategray')
    snippet_plot.plot(mean, color='red')
    plt.xticks([0, 10, 30], [-10, 0, 20])


def plot_spikes_for_channel_centered(grid, spike_data, cluster_id, channel, snippet_column_name, mean_color='red'):
    cluster_df = spike_data[(spike_data.cluster_id == cluster_id)]  # dataframe for that cluster
    snippet_plot = plt.subplot(grid[int(channel / 2), channel % 2])
    plot_utility.style_plot(snippet_plot)
    if len(cluster_df[snippet_column_name].iloc[0]) > 0:
        max_channel = cluster_df['primary_channel'].iloc[0]
        sd = np.std(cluster_df['random_snippets'].iloc[0][max_channel - 1, :, :] * -1)
        highest_value = np.mean(cluster_df['random_snippets'].iloc[0][max_channel - 1, :, :] * -1) + (sd * 4)
        lowest_value = np.mean(cluster_df['random_snippets'].iloc[0][max_channel - 1, :, :] * -1) - (sd * 4)
        plt.ylim(lowest_value - 10, highest_value + 30)
        snippet_plot.plot(cluster_df[snippet_column_name].iloc[0][channel, :, :] * -1, color='lightslategray')
        snippet_plot.plot(np.mean(cluster_df[snippet_column_name].iloc[0][channel, :, :], 1) * -1, color=mean_color)
    plt.xticks([0, 30], [0, 1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Time (ms)', fontsize=14)
    plt.ylabel('Voltage (µV)', fontsize=14)


def plot_waveforms(spike_data, output_path):
    print('I will plot the waveform shapes for each cluster.')
    save_path = output_path + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        fig = plt.figure(figsize=(5, 5))
        plt.suptitle("Spike waveforms", fontsize=24)
        grid = plt.GridSpec(2, 2, wspace=1, hspace=0.5)
        for channel in range(4):
            plot_spikes_for_channel_centered(grid, spike_data, cluster_id, channel, 'random_snippets')

        plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '_waveforms.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_firing_properties(spike_data, output_path):
    plot_waveforms(spike_data, output_path)
    plot_spike_histogram(spike_data, output_path)
    plot_autocorrelograms(spike_data, output_path)
    return