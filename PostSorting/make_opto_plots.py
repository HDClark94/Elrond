import array_utility
import os
import matplotlib.pylab as plt
import math
import numpy as np
import pandas as pd
import plot_utility
import PostSorting.parameters
import PostSorting.make_plots
import scipy.ndimage
import matplotlib.image as mpimg


def make_folder_for_figures(output_path):
    save_path = output_path + '/Figures/opto_stimulation'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    return save_path


def get_first_spikes(cluster_rows, stimulation_start, sampling_rate, latency_window_ms=10):
    '''
    Find first spikes after the light pulse in a 10ms window
    :param cluster_rows: Opto stimulation trials corresponding to cluster (binary array)
    :param stimulation_start: Index (in samplin rate) where the stimulation starts in the peristimulus array)
    :param sampling_rate: Sampling rate of electrophysiology data
    :param latency_window_ms: The time window used to calculate latencies in ms (spikes outside this are not included)
    :return:
    '''

    latency_window = latency_window_ms * sampling_rate / 1000  # this is the latency window in sampling points
    events_after_stimulation = cluster_rows[
        cluster_rows.columns[int(stimulation_start):int(stimulation_start + latency_window)]]
    # events_after_stimulation = cluster_rows[cluster_rows.columns[int(stimulation_start):]]
    spikes = np.array(events_after_stimulation).astype(int) == 1
    first_spikes = spikes.cumsum(axis=1).cumsum(axis=1) == 1
    zeros_left = np.zeros((spikes.shape[0], int(stimulation_start - 1)))
    first_spikes = np.hstack((zeros_left, first_spikes))
    zeros_right = np.zeros((spikes.shape[0], int(cluster_rows.shape[1] - stimulation_start - sampling_rate / 100)))
    first_spikes = np.hstack((first_spikes, zeros_right))
    sample_times_firsts = np.argwhere(first_spikes)[:, 1]
    trial_numbers_firsts = np.argwhere(first_spikes)[:, 0]
    return sample_times_firsts, trial_numbers_firsts


def plot_spikes_around_light(ax, cluster_rows, sampling_rate, light_pulse_duration, latency_window_ms):
    sample_times = np.argwhere(np.array(cluster_rows).astype(int) == 1)[:, 1]
    trial_numbers = np.argwhere(np.array(cluster_rows).astype(int) == 1)[:, 0]
    stimulation_start = cluster_rows.shape[1] / 2  # the peristimulus array is made so the pulse starts in the middle
    stimulation_end = cluster_rows.shape[1] / 2 + light_pulse_duration

    ax.axvspan(stimulation_start, stimulation_end, 0, cluster_rows.shape[0], alpha=0.5, color='lightblue')
    ax.vlines(x=sample_times, ymin=trial_numbers, ymax=(trial_numbers + 1), color='black', zorder=2, linewidth=3)
    sample_times_firsts, trial_numbers_firsts = get_first_spikes(cluster_rows, stimulation_start, sampling_rate,
                                                                 latency_window_ms)
    ax.vlines(x=sample_times_firsts, ymin=trial_numbers_firsts, ymax=(trial_numbers_firsts + 1), color='red', zorder=2,
              linewidth=3)


def format_peristimulus_plot():
    """
    Add axis labels and set size of figures.
    """
    plt.cla()
    peristimulus_figure, ax = plt.subplots()
    peristimulus_figure.set_size_inches(5, 5, forward=True)
    plt.xlabel('Time (sampling points)', fontsize=16)
    plt.ylabel('Trial (sampling points)', fontsize=16)
    return peristimulus_figure, ax


def get_binary_peristimulus_data_for_cluster(peristimulus_spikes: pd.DataFrame, cluster: str):
    """
    :param peristimulus_spikes: Data frame with firing times of all clusters around the stimulus
    the first two columns the session and cluster ids respectively and the rest of the columns correspond to
    sampling points before and after the stimulus. 0s mean no spike and 1s mean spike at a given point
    :param cluster: cluster id
    :return: rows of data frame that correspond to cluster and only the columns that contain the binary spike data
    """

    cluster_rows_boolean = peristimulus_spikes.cluster_id.astype(int) == int(cluster)
    cluster_rows_annotated = peristimulus_spikes[cluster_rows_boolean]
    cluster_rows = cluster_rows_annotated.iloc[:, 2:]
    return cluster_rows


def plot_peristimulus_raster_for_cluster(peristimulus_spikes, cluster, session, sampling_rate, light_pulse_duration,
                                         latency_window_ms, save_path):
    cluster_rows = get_binary_peristimulus_data_for_cluster(peristimulus_spikes, cluster)
    peristimulus_figure, ax = format_peristimulus_plot()
    plot_spikes_around_light(ax, cluster_rows, sampling_rate, light_pulse_duration, latency_window_ms)
    plt.ylim(0, cluster_rows.shape[0])
    plt.xlim(0, cluster_rows.shape[1])
    plt.savefig(save_path + '/' + session.iloc[0] + '_' + str(cluster) + '_peristimulus_raster.png', dpi=300)
    plt.close()


# do not use this on data from more than one session
def plot_peristimulus_raster(peristimulus_spikes: pd.DataFrame, output_path: str, sampling_rate: int,
                             light_pulse_duration: int, latency_window_ms: int):
    """
    PLots spike raster from light stimulation trials around the light. The plot assumes that the stimulation
    starts in the middle of the peristimulus_spikes array.
    :param peristimulus_spikes: Data frame with firing times of all clusters around the stimulus
    the first two columns the session and cluster ids respectively and the rest of the columns correspond to
    sampling points before and after the stimulus. 0s mean no spike and 1s mean spike at a given point
    :param output_path: fist half of the path where the plot is saved
    :param sampling_rate: sampling rate of electrophysiology data
    :param light_pulse_duration: duration of light pulse (ms)
    :param latency_window_ms: time window where spikes are considered evoked for a given trial
    """

    # make sure it's a single session
    assert len(peristimulus_spikes.groupby('session_id')['session_id'].nunique()) == 1
    save_path = make_folder_for_figures(output_path)
    cluster_ids = np.unique(peristimulus_spikes.cluster_id)
    for cluster in cluster_ids:
        plot_peristimulus_raster_for_cluster(peristimulus_spikes, cluster, peristimulus_spikes.session_id, sampling_rate, light_pulse_duration,
                                             latency_window_ms, save_path)


def get_latencies_for_cluster(spatial_firing, cluster_id):
    cluster = spatial_firing[spatial_firing.cluster_id == int(cluster_id)]
    latencies_mean = np.round(cluster.opto_latencies_mean_ms, 2)
    latencies_sd = np.round(cluster.opto_latencies_sd_ms, 2)
    if len(latencies_mean) > 0:
        return pd.to_numeric(latencies_mean).iloc[0], pd.to_numeric(latencies_sd).iloc[0]
    else:
        return pd.to_numeric(latencies_mean), pd.to_numeric(latencies_sd)


def make_peristimulus_histogram_for_cluster(spatial_firing, peristimulus_spikes, cluster, session, light_pulse_duration,
                                            save_path):
    cluster_rows = get_binary_peristimulus_data_for_cluster(peristimulus_spikes, cluster)
    cluster_rows = cluster_rows.astype(int).to_numpy()
    peristimulus_figure, ax = format_peristimulus_plot()
    number_of_spikes_per_sampling_point = np.array(np.sum(cluster_rows, axis=0))
    stimulation_start = cluster_rows.shape[1] / 2  # stimulus pulse starts in the middle of the array
    stimulation_end = cluster_rows.shape[1] / 2 + light_pulse_duration
    latencies_mean, latencies_sd = get_latencies_for_cluster(spatial_firing, cluster)
    ax.axvspan(stimulation_start, stimulation_end, 0, np.max(number_of_spikes_per_sampling_point), alpha=0.5,
               color='lightblue')
    # convert to indices so we can make histogram
    spike_indices = np.where(cluster_rows.flatten() == 1)[0] % len(number_of_spikes_per_sampling_point)
    plt.hist(spike_indices, color='grey', alpha=0.5, bins=50)
    plt.xlim(0, len(number_of_spikes_per_sampling_point))
    plt.title('Mean latency: ' + str(latencies_mean) + ' ms, sd = ' + str(latencies_sd))
    plt.savefig(save_path + '/' + session.iloc[0] + '_' + str(cluster) + '_peristimulus_histogram.png', dpi=300)
    plt.close()


def plot_peristimulus_histogram(spatial_firing: pd.DataFrame, peristimulus_spikes: pd.DataFrame, output_path: str,
                                light_pulse_duration: int):
    """
    PLots histogram of spikes from light stimulation trials around the light. The plot assumes that the stimulation
    starts in the middle of the peristimulus_spikes array.
    :param spatial_firing: Data frame with firing data for each cluster
    :param peristimulus_spikes: Data frame with firing times of all clusters around the stimulus
    :param output_path: fist half of the path where the plot is saved
    the first two columns the session and cluster ids respectively and the rest of the columns correspond to
    sampling points before and after the stimulus. 0s mean no spike and 1s mean spike at a given point
    :param light_pulse_duration: duration of light pulse (ms)
    """
    assert len(peristimulus_spikes.groupby('session_id')['session_id'].nunique()) == 1
    save_path = make_folder_for_figures(output_path)
    cluster_ids = np.unique(peristimulus_spikes.cluster_id)
    for cluster in cluster_ids:
        make_peristimulus_histogram_for_cluster(spatial_firing, peristimulus_spikes, cluster, peristimulus_spikes.session_id, light_pulse_duration,
                                                save_path)


def plot_waveforms_opto(spike_data, output_path, snippets_column_name='random_snippets_opto', title='Random snippets'):
    if snippets_column_name in spike_data:
        print('I will plot the waveform shapes for each cluster for opto_tagging data.')
        save_path = output_path + '/Figures/opto_stimulation'
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)

        for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
            cluster_df = spike_data[(spike_data.cluster_id == cluster_id)]  # dataframe for that cluster

            max_channel = cluster_df['primary_channel'].iloc[0]
            fig = plt.figure(figsize=(5, 5))
            plt.suptitle(title, fontsize=24)
            grid = plt.GridSpec(2, 2, wspace=0.5, hspace=0.5)
            for channel in range(4):
                PostSorting.make_plots.plot_spikes_for_channel_centered(grid, spike_data, cluster_id, channel,
                                                                        snippets_column_name)

            plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_' + str(
                cluster_id) + '_' + snippets_column_name + '.png', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()


def make_combined_opto_plot(spatial_firing, output_path):
    print('I will make the combined images for opto stimulation analysis results now.')
    save_path = output_path + '/Figures/opto_plots_combined'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.close('all')
    figures_path = output_path + '/Figures/'
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)]  # data frame for that cluster

        waveforms_cell_all = figures_path + 'firing_properties/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '_waveforms.png'
        waveforms_opto_random = figures_path + 'opto_stimulation/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '_random_snippets_opto.png'
        waveforms_first_spikes = figures_path + 'opto_stimulation/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '_random_first_spike_snippets_opto.png'
        peristimulus_raster = figures_path + 'opto_stimulation/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '_peristimulus_raster.png'
        peristimulus_histogram = figures_path + 'opto_stimulation/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '_peristimulus_histogram.png'

        number_of_rows = 2
        number_of_columns = 3
        grid = plt.GridSpec(number_of_rows, number_of_columns, wspace=0.2, hspace=0.2)
        plt.suptitle('Light responses')

        if os.path.exists(waveforms_cell_all):
            waves_all = mpimg.imread(waveforms_cell_all)
            waves_all_plot = plt.subplot(grid[0, 0])
            waves_all_plot.axis('off')
            waves_all_plot.imshow(waves_all)
        if os.path.exists(waveforms_opto_random):
            waves_opto = mpimg.imread(waveforms_opto_random)
            waves_opto_plot = plt.subplot(grid[0, 1])
            waves_opto_plot.axis('off')
            waves_opto_plot.imshow(waves_opto)
        if os.path.exists(waveforms_first_spikes):
            first_spks = mpimg.imread(waveforms_first_spikes)
            first_spks_plot = plt.subplot(grid[0, 2])
            first_spks_plot.axis('off')
            first_spks_plot.imshow(first_spks)
        if os.path.exists(peristimulus_raster):
            peristim_raster = mpimg.imread(peristimulus_raster)
            pristim_plot = plt.subplot(grid[1, 0])
            pristim_plot.axis('off')
            pristim_plot.imshow(peristim_raster)
        if os.path.exists(peristimulus_histogram):
            peristim_hist = mpimg.imread(peristimulus_histogram)
            pristim_plot = plt.subplot(grid[1, 1])
            pristim_plot.axis('off')
            pristim_plot.imshow(peristim_hist)

        plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '.png', dpi=1000)
        # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '.pdf', dpi=1000)
        plt.close()


def make_optogenetics_plots(spatial_firing: pd.DataFrame, output_path: str, sampling_rate: int):
    """
    :param spatial_firing: data frame where each row corresponds to a cluster
    :param output_path: output folder to save figures in (usually /MountainSort)
    :param sampling_rate: sampling rate of electrophysiology data
    """

    peristimulus_spikes_path = output_path + '/DataFrames/peristimulus_spikes.pkl'
    opto_parameters_path = output_path + '/DataFrames/opto_parameters.pkl'
    if os.path.exists(peristimulus_spikes_path):
        if os.path.exists(opto_parameters_path):
            opto_parameters = pd.read_pickle(opto_parameters_path)
            light_pulse_duration = opto_parameters.duration.iloc[0] * sampling_rate / 1000
            latency_window_ms = opto_parameters.first_spike_latency_ms.iloc[0]
        else:
            print('There is no metadata saved for optical stimulation. I will assume the pulses are 3 ms and that '
                  'the latencies should be calculated in a 10ms window.')
            light_pulse_duration = 90
            latency_window_ms = 10

        # binary array containing light stimulation trials in each row (0 means no spike 1 means spike at a sampling point)
        peristimulus_spikes = pd.read_pickle(peristimulus_spikes_path)
        plot_peristimulus_raster(peristimulus_spikes, output_path, sampling_rate, light_pulse_duration=light_pulse_duration,
                                 latency_window_ms=latency_window_ms)
        plot_peristimulus_histogram(spatial_firing, peristimulus_spikes, output_path, light_pulse_duration=light_pulse_duration)
        plot_waveforms_opto(spatial_firing, output_path, snippets_column_name='random_snippets_opto', title='During opto-tagging')
        plot_waveforms_opto(spatial_firing, output_path, snippets_column_name='random_first_spike_snippets_opto', title='First spikes after light')
        make_combined_opto_plot(spatial_firing, output_path)


def main():
    path = 'C:/Users/s1466507/Documents/Work/opto/M2_2021-02-17_18-07-42_of/MountainSort/DataFrames/peristimulus_spikes.pkl'
    peristimulus_spikes = pd.read_pickle(path)
    path = 'C:/Users/s1466507/Documents/Work/opto/M2_2021-02-17_18-07-42_of/MountainSort/DataFrames/spatial_firing.pkl'
    spatial_firing = pd.read_pickle(path)
    sampling_rate = 30000
    output_path = ('C:/Users/s1466507/Documents/Work/opto/M2_2021-02-17_18-07-42_of/MountainSort/')
    plot_peristimulus_raster(peristimulus_spikes, output_path, sampling_rate, light_pulse_duration=90,
                             latency_window_ms=10)
    plot_peristimulus_histogram(spatial_firing, peristimulus_spikes, output_path, light_pulse_duration=90)


if __name__ == '__main__':
    main()
