import glob
import numpy as np
import os
import pandas as pd
from scipy import stats
import shutil
import sys
from statsmodels.sandbox.stats.multicomp import multipletests
import threading
import OverallAnalysis.false_positives
import OverallAnalysis.folder_path_settings
import data_frame_utility
import matplotlib.pylab as plt

server_path_mouse = OverallAnalysis.folder_path_settings.get_server_path_mouse()
server_path_rat = OverallAnalysis.folder_path_settings.get_server_path_rat()


def add_combined_id_to_df(df_all_mice):
    animal_ids = [session_id.split('_')[0] for session_id in df_all_mice.session_id.values]
    dates = [session_id.split('_')[1] for session_id in df_all_mice.session_id.values]
    tetrode = df_all_mice.tetrode.values
    cluster = df_all_mice.cluster_id.values

    combined_ids = []
    for cell in range(len(df_all_mice)):
        id = animal_ids[cell] +  '-' + dates[cell] + '-Tetrode-' + str(tetrode[cell]) + '-Cluster-' + str(cluster[cell])
        combined_ids.append(id)
    df_all_mice['false_positive_id'] = combined_ids
    return df_all_mice


def format_bar_chart(ax):
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel('Head direction [deg]', fontsize=30)
    ax.set_ylabel('Frequency [Hz]', fontsize=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    return ax


def add_mean_and_std_to_field_df(field_data, sampling_rate_video, number_of_bins=20):
    fields_means = []
    fields_stdevs = []
    real_data_hz_all_fields = []
    time_spent_in_bins_all = []
    field_histograms_hz_all = []
    for index, field in field_data.iterrows():
        field_histograms = field['shuffled_data']
        field_spikes_hd = field['hd_in_field_spikes']  # real hd when the cell fired
        field_session_hd = field['hd_in_field_session']  # hd from the whole session in field
        time_spent_in_bins = np.histogram(field_session_hd, bins=number_of_bins)[0]
        time_spent_in_bins_all.append(time_spent_in_bins)
        field_histograms_hz = field_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        field_histograms_hz_all.append(field_histograms_hz)
        mean_shuffled = np.mean(field_histograms_hz, axis=0)
        fields_means.append(mean_shuffled)
        std_shuffled = np.std(field_histograms_hz, axis=0)
        fields_stdevs.append(std_shuffled)

        real_data_hz = np.histogram(field_spikes_hd, bins=20)[0] * sampling_rate_video / time_spent_in_bins
        real_data_hz_all_fields.append(real_data_hz)
    field_data['shuffled_means'] = fields_means
    field_data['shuffled_std'] = fields_stdevs
    field_data['hd_histogram_real_data'] = real_data_hz_all_fields
    field_data['time_spent_in_bins'] = time_spent_in_bins_all
    field_data['field_histograms_hz'] = field_histograms_hz_all
    return field_data


def add_percentile_values_to_df(field_data, sampling_rate_video, number_of_bins=20):
    field_percentile_values_95_all = []
    field_percentile_values_5_all = []
    error_bar_up_all = []
    error_bar_down_all = []
    for index, field in field_data.iterrows():
        field_histograms = field['shuffled_data']
        field_session_hd = field['hd_in_field_session']  # hd from the whole session in field
        time_spent_in_bins = np.histogram(field_session_hd, bins=number_of_bins)[0]
        field_histograms_hz = field_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        percentile_value_shuffled_95 = np.percentile(field_histograms_hz, 95, axis=0)
        field_percentile_values_95_all.append(percentile_value_shuffled_95)
        percentile_value_shuffled_5 = np.percentile(field_histograms_hz, 5, axis=0)
        field_percentile_values_5_all.append(percentile_value_shuffled_5)
        error_bar_up = percentile_value_shuffled_95 - field.shuffled_means
        error_bar_down = field.shuffled_means - percentile_value_shuffled_5
        error_bar_up_all.append(error_bar_up)
        error_bar_down_all.append(error_bar_down)
    field_data['shuffled_percentile_threshold_95'] = field_percentile_values_95_all
    field_data['shuffled_percentile_threshold_5'] = field_percentile_values_5_all
    field_data['error_bar_95'] = error_bar_up_all
    field_data['error_bar_5'] = error_bar_down_all
    return field_data


# test whether real and shuffled data differ and add results (true/false for each bin) and number of diffs to data frame
def test_if_real_hd_differs_from_shuffled(field_data):
    real_and_shuffled_data_differ_bin = []
    number_of_diff_bins = []
    for index, field in field_data.iterrows():
        diff_field = (field.shuffled_percentile_threshold_95 < field.hd_histogram_real_data) + (field.shuffled_percentile_threshold_5 > field.hd_histogram_real_data)  # this is a pairwise OR on the binary arrays
        number_of_diffs = diff_field.sum()
        real_and_shuffled_data_differ_bin.append(diff_field)
        number_of_diff_bins.append(number_of_diffs)
    field_data['real_and_shuffled_data_differ_bin'] = real_and_shuffled_data_differ_bin
    field_data['number_of_different_bins'] = number_of_diff_bins
    return field_data


# this uses the p values that are based on the position of the real data relative to shuffled (corrected_
def count_number_of_significantly_different_bars_per_field(field_data, significance_level=95, type='bh'):
    number_of_significant_p_values = []
    false_positive_ratio = (100 - significance_level) / 100
    for index, field in field_data.iterrows():
        # count significant p values
        if type == 'bh':
            number_of_significant_p_values_field = (field.p_values_corrected_bars_bh < false_positive_ratio).sum()
            number_of_significant_p_values.append(number_of_significant_p_values_field)
        if type == 'holm':
            number_of_significant_p_values_field = (field.p_values_corrected_bars_holm < false_positive_ratio).sum()
            number_of_significant_p_values.append(number_of_significant_p_values_field)
    field_name = 'number_of_different_bins_' + type
    field_data[field_name] = number_of_significant_p_values
    return field_data


# this is to find the null distribution of number of rejected null hypothesis based on the shuffled data
def test_if_shuffle_differs_from_other_shuffles(field_data):
    number_of_shuffles = len(field_data.shuffled_data[0])
    rejected_bins_all_shuffles = []
    for index, field in field_data.iterrows():
        rejects_field = np.empty(number_of_shuffles)
        rejects_field[:] = np.nan
        for shuffle in range(number_of_shuffles):
            diff_field = (field.shuffled_percentile_threshold_95 < field.field_histograms_hz[shuffle]) + (field.shuffled_percentile_threshold_5 > field.field_histograms_hz[shuffle])  # this is a pairwise OR on the binary arrays
            number_of_diffs = diff_field.sum()
            rejects_field[shuffle] = number_of_diffs
        rejected_bins_all_shuffles.append(rejects_field)
    field_data['number_of_different_bins_shuffled'] = rejected_bins_all_shuffles
    return field_data


# this is to find the null distribution of number of rejected null hypothesis based on the shuffled data
# perform B/H analysis on each shuffle and count rejects
def test_if_shuffle_differs_from_other_shuffles_corrected_p_values(field_data, sampling_rate_video, number_of_bars=20):
    number_of_shuffles = len(field_data.shuffled_data[0])
    rejected_bins_all_shuffles = []
    for index, field in field_data.iterrows():
        field_histograms = field['shuffled_data']
        field_session_hd = field['hd_in_field_session']  # hd from the whole session in field
        time_spent_in_bins = np.histogram(field_session_hd, bins=number_of_bars)[0]
        shuffled_data_normalized = field_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        rejects_field = np.empty(number_of_shuffles)
        rejects_field[:] = np.nan
        percentile_observed_data_bars = []
        for shuffle in range(number_of_shuffles):
            percentiles_of_observed_bars = np.empty(number_of_bars)
            percentiles_of_observed_bars[:] = np.nan
            for bar in range(number_of_bars):
                observed_data = shuffled_data_normalized[shuffle][bar]
                shuffled_data = shuffled_data_normalized[:, bar]
                percentile_of_observed_data = stats.percentileofscore(shuffled_data, observed_data)
                percentiles_of_observed_bars[bar] = percentile_of_observed_data
            percentile_observed_data_bars.append(percentiles_of_observed_bars)  # percentile of shuffle relative to all other shuffles
            # convert percentile to p value
            percentiles_of_observed_bars[percentiles_of_observed_bars > 50] = 100 - percentiles_of_observed_bars[percentiles_of_observed_bars > 50]
            # correct p values (B/H)
            reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(percentiles_of_observed_bars, alpha=0.05, method='fdr_bh')
            # count significant bars and put this number in df
            number_of_rejects = reject.sum()
            rejects_field[shuffle] = number_of_rejects
        rejected_bins_all_shuffles.append(rejects_field)
    field_data['number_of_different_bins_shuffled_corrected_p'] = rejected_bins_all_shuffles
    return field_data


# calculate percentile of real data relative to shuffled for each bar
def calculate_percentile_of_observed_data(field_data, sampling_rate_video, number_of_bars=20):
    percentile_observed_data_bars = []
    for index, field in field_data.iterrows():
        field_histograms = field['shuffled_data']
        field_session_hd = field['hd_in_field_session']  # hd from the whole session in field
        time_spent_in_bins = np.histogram(field_session_hd, bins=number_of_bars)[0]
        shuffled_data_normalized = field_histograms * sampling_rate_video / time_spent_in_bins  # sampling rate is 30Hz for movement data
        percentiles_of_observed_bars = np.empty(number_of_bars)
        percentiles_of_observed_bars[:] = np.nan
        for bar in range(number_of_bars):
            observed_data = field.hd_histogram_real_data[bar]
            shuffled_data = shuffled_data_normalized[:, bar]
            percentile_of_observed_data = stats.percentileofscore(shuffled_data, observed_data)
            percentiles_of_observed_bars[bar] = percentile_of_observed_data
        percentile_observed_data_bars.append(percentiles_of_observed_bars)
    field_data['percentile_of_observed_data'] = percentile_observed_data_bars
    return field_data


#  convert percentile to p value by subtracting the percentile from 100 when it is > than 50
def convert_percentile_to_p_value(field_data):
    p_values = []
    for index, field in field_data.iterrows():
        percentile_values = field.percentile_of_observed_data
        percentile_values[percentile_values > 50] = 100 - percentile_values[percentile_values > 50]
        p_values.append(percentile_values)
    field_data['shuffle_p_values'] = p_values
    return field_data


# perform Benjamini/Hochberg correction on p values calculated from the percentile of observed data relative to shuffled
def calculate_corrected_p_values(field_data, type='bh'):
    corrected_p_values = []
    for index, field in field_data.iterrows():
        p_values = field.shuffle_p_values
        if type == 'bh':
            reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_values, alpha=0.05, method='fdr_bh')
            corrected_p_values.append(pvals_corrected)
        if type == 'holm':
            reject, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_values, alpha=0.05, method='holm')
            corrected_p_values.append(pvals_corrected)

    field_name = 'p_values_corrected_bars_' + type
    field_data[field_name] = corrected_p_values
    return field_data


def plot_bar_chart_for_fields(field_data, sampling_rate_video, path):
    for index, field in field_data.iterrows():
        mean = field['shuffled_means']
        std = field['shuffled_std']
        field_spikes_hd = field['hd_in_field_spikes']
        time_spent_in_bins = field['time_spent_in_bins']
        field_histograms_hz = field['field_histograms_hz']
        x_pos = np.arange(field_histograms_hz.shape[1])
        fig, ax = plt.subplots()
        ax = format_bar_chart(ax)
        ax.bar(x_pos, mean, yerr=std*2, align='center', alpha=0.7, color='black', ecolor='grey', capsize=10)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        real_data_hz = np.histogram(field_spikes_hd, bins=20)[0] * sampling_rate_video / time_spent_in_bins
        plt.scatter(x_pos, real_data_hz, marker='o', color='red', s=40)
        plt.savefig(path + 'shuffle_analysis/' + str(field['cluster_id']) + '_field_' + str(index) + '_SD')
        plt.close()


def plot_bar_chart_for_fields_percentile_error_bar(field_data, sampling_rate_video, path):
    for index, field in field_data.iterrows():
        mean = field['shuffled_means']
        percentile_95 = field['error_bar_95']
        percentile_5 = field['error_bar_5']
        field_spikes_hd = field['hd_in_field_spikes']
        time_spent_in_bins = field['time_spent_in_bins']
        field_histograms_hz = field['field_histograms_hz']
        x_pos = np.arange(field_histograms_hz.shape[1])
        fig, ax = plt.subplots()
        ax = format_bar_chart(ax)
        ax.errorbar(x_pos, mean, yerr=[percentile_5, percentile_95], alpha=0.7, color='black', ecolor='grey', capsize=10, fmt='o', markersize=10)
        # ax.bar(x_pos, mean, yerr=[percentile_5, percentile_95], align='center', alpha=0.7, color='black', ecolor='grey', capsize=10)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        real_data_hz = np.histogram(field_spikes_hd, bins=20)[0] * sampling_rate_video / time_spent_in_bins
        plt.scatter(x_pos, real_data_hz, marker='o', color='red', s=40)
        plt.savefig(path + 'shuffle_analysis/' + str(field['cluster_id']) + '_field_' + str(index) + '_percentile')
        plt.close()


# the results of these are added to field_data so it can be combined from all cells later (see load_data_frames and shuffle_field_analysis_all_mice)
def analyze_shuffled_data(field_data, save_path, sampling_rate_video, number_of_bins=20):
    field_data = add_mean_and_std_to_field_df(field_data, sampling_rate_video, number_of_bins)
    field_data = add_percentile_values_to_df(field_data, sampling_rate_video, number_of_bins=20)
    field_data = test_if_real_hd_differs_from_shuffled(field_data)  # is the observed data within 95th percentile of the shuffled?
    field_data = test_if_shuffle_differs_from_other_shuffles(field_data)

    field_data = calculate_percentile_of_observed_data(field_data, sampling_rate_video, number_of_bins)  # this is relative to shuffled data
    # field_data = calculate_percentile_of_shuffled_data(field_data, number_of_bars=20)
    field_data = convert_percentile_to_p_value(field_data)  # this is needed to make it 2 tailed so diffs are picked up both ways
    field_data = calculate_corrected_p_values(field_data, type='bh')  # BH correction on p values from previous function
    field_data = calculate_corrected_p_values(field_data, type='holm')  # Holm correction on p values from previous function
    field_data = count_number_of_significantly_different_bars_per_field(field_data, significance_level=95, type='bh')
    field_data = count_number_of_significantly_different_bars_per_field(field_data, significance_level=95, type='holm')
    field_data = test_if_shuffle_differs_from_other_shuffles_corrected_p_values(field_data, sampling_rate_video, number_of_bars=20)
    plot_bar_chart_for_fields(field_data, sampling_rate_video, save_path)
    plot_bar_chart_for_fields_percentile_error_bar(field_data, sampling_rate_video, save_path)
    return field_data


def get_random_indices_for_shuffle(field, number_of_times_to_shuffle):
    number_of_spikes_in_field = field['number_of_spikes_in_field']
    time_spent_in_field = field['time_spent_in_field']
    shuffle_indices = np.random.randint(0, time_spent_in_field, size=(number_of_times_to_shuffle, number_of_spikes_in_field))
    return shuffle_indices


# add shuffled data to data frame as a new column for each field
def shuffle_field_data(field_data, path, number_of_bins, number_of_times_to_shuffle=1000):
    if os.path.exists(path + 'shuffle_analysis') is True:
        shutil.rmtree(path + 'shuffle_analysis')
    os.makedirs(path + 'shuffle_analysis')
    field_histograms_all = []
    for index, field in field_data.iterrows():
        print('I will shuffle data in the fields.')
        field_histograms = np.zeros((number_of_times_to_shuffle, number_of_bins))
        shuffle_indices = get_random_indices_for_shuffle(field, number_of_times_to_shuffle)
        for shuffle in range(number_of_times_to_shuffle):
            shuffled_hd = field['hd_in_field_session'][shuffle_indices[shuffle]]
            hist, bin_edges = np.histogram(shuffled_hd, bins=number_of_bins, range=(0, 6.28))  # from 0 to 2pi
            field_histograms[shuffle, :] = hist
        field_histograms_all.append(field_histograms)
    print(path)
    field_data['shuffled_data'] = field_histograms_all
    return field_data


# perform shuffle analysis for all animals and save data frames on server. this will later be loaded and combined
def process_recordings(server_path, sampling_rate_video, spike_sorter='/MountainSort', redo_existing=True):
    if os.path.exists(server_path):
        print('I see the server.')
    for recording_folder in glob.glob(server_path + '*'):
        if os.path.isdir(recording_folder):
            spike_data_frame_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
            position_data_frame_path = recording_folder + spike_sorter + '/DataFrames/position.pkl'
            shuffled_data_frame_path = recording_folder + spike_sorter + '/DataFrames/shuffled_fields.pkl'
            if os.path.exists(spike_data_frame_path):
                print('I found a firing data frame.')
                if redo_existing is False:
                    if os.path.exists(shuffled_data_frame_path):
                        print('This was shuffled earlier.')
                        print(recording_folder)
                        continue
                spatial_firing = pd.read_pickle(spike_data_frame_path)
                position_data = pd.read_pickle(position_data_frame_path)
                field_df = data_frame_utility.get_field_data_frame(spatial_firing, position_data)
                if not field_df.empty:
                    field_df = shuffle_field_data(field_df, recording_folder + '/MountainSort/', number_of_bins=20, number_of_times_to_shuffle=1000)
                    field_df = analyze_shuffled_data(field_df, recording_folder + '/MountainSort/', sampling_rate_video, number_of_bins=20)
                    try:
                        field_df.to_pickle(recording_folder + spike_sorter + '/DataFrames/shuffled_fields.pkl')
                        print('I finished analyzing ' + recording_folder)
                    except OSError as error:
                        print('ERROR I failed to analyze ' + recording_folder)
                        print(error)


# this is to test functions without accessing the server
def local_data_test():
    local_path = OverallAnalysis.folder_path_settings.get_local_test_recording_path()
    spatial_firing = pd.read_pickle(local_path + '/DataFrames/spatial_firing.pkl')
    position_data = pd.read_pickle(local_path + '/DataFrames/position.pkl')

    field_df = data_frame_utility.get_field_data_frame(spatial_firing, position_data)
    field_df = shuffle_field_data(field_df, local_path, number_of_bins=20, number_of_times_to_shuffle=1000)
    field_df = analyze_shuffled_data(field_df, 30, local_path, number_of_bins=20)
    field_df.to_pickle(local_path + 'shuffle_analysis/shuffled_fields.pkl')


def main():
    sys.setrecursionlimit(100000)
    threading.stack_size(200000000)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    process_recordings(server_path_mouse, 30, redo_existing=True)
    process_recordings(server_path_rat, 50, spike_sorter='', redo_existing=True)
    # local_data_test()


if __name__ == '__main__':
    main()
