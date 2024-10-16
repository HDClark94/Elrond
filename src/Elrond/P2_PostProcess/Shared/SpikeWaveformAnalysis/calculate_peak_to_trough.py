import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd

"""
## the following code calculates the distance between the peak and trough of snippets
1. extract snippets
2. calculate mean snippet
3. identify peak and trough position and calculate distance
4. insert into dataframe
"""


def remove_outlier_waveforms(all_waveforms, max_deviations=3):
    # remove snippets that have data points > 3 standard dev away from mean
    mean = all_waveforms.mean(axis=1)
    sd = all_waveforms.std(axis=1)
    distance_from_mean = all_waveforms.T - mean
    outliers = np.sum(distance_from_mean > max_deviations * sd, axis=1) > 0
    return all_waveforms[:, ~outliers]


def get_peak_and_trough_positions(mean_waveform):
    peak = np.argmax(np.absolute(mean_waveform))
    if peak < len(mean_waveform):
        if mean_waveform[peak] < 0:
            trough = np.argmax(mean_waveform[peak:]) + peak
        else:  # this happens when the spike is 'upside down' because if the tetrode position relative to the neuron
            trough = np.argmin(mean_waveform[peak:]) + peak
    else:
        trough = np.argmin(mean_waveform)
    return peak, trough


def add_trough_to_peak_to_df(spatial_firing):
    peak_to_trough = []
    snippet_peak_position = []
    snippet_trough_position = []
    for index, cell in spatial_firing.iterrows():
        primary_channel = cell.primary_channel - 1
        all_waveforms_with_noise = cell.random_snippets[primary_channel]
        all_waveforms = remove_outlier_waveforms(all_waveforms_with_noise)
        mean_waveform = all_waveforms.mean(axis=1)
        peak, trough = get_peak_and_trough_positions(mean_waveform)
        snippet_peak_position.append(peak)
        snippet_trough_position.append(trough)
        peak_to_trough.append(np.abs(peak-trough))

    spatial_firing['peak_to_trough'] = peak_to_trough
    spatial_firing['snippet_peak_position'] = snippet_peak_position
    spatial_firing['snippet_trough_position'] = snippet_trough_position
    return spatial_firing


def visualize_peak_to_trough_detection(spatial_firing):
    for index, cell in spatial_firing.iterrows():
        primary_channel = cell.primary_channel - 1
        all_waveforms_with_noise = cell.random_snippets[primary_channel]
        all_waveforms = remove_outlier_waveforms(all_waveforms_with_noise)
        mean_waveform = all_waveforms.mean(axis=1)
        # plt.plot(all_waveforms_with_noise, color='grey', alpha=0.6)
        plt.plot(all_waveforms, color='skyblue', alpha=0.8)
        plt.plot(mean_waveform, linewidth=3, color='navy')
        plt.title(str(np.round(cell.mean_firing_rate,2)) + ' Hz')
        plt.axvline(cell.snippet_peak_position, color='red')
        plt.axvline(cell.snippet_trough_position, color='red')
        plt.show()


def analyse_waveform_shapes(recording_folder_path):
    print('Calculate peak to trough distance for each cell.')
    print(recording_folder_path)
    spatial_firing_path = recording_folder_path + '/MountainSort/DataFrames/spatial_firing.pkl'
    if os.path.exists(spatial_firing_path):
        spatial_firing = pd.read_pickle(spatial_firing_path)
        spatial_firing = add_trough_to_peak_to_df(spatial_firing)
        # visualize_peak_to_trough_detection(spatial_firing)
        spatial_firing.to_pickle(recording_folder_path + '/MountainSort/DataFrames/spatial_firing.pkl')

    else:
        print('There is no spatial firing data for this recording: ' + recording_folder_path)
        return False


def process_recordings(recording_list):
    for recording in recording_list:
        analyse_waveform_shapes(recording)
    print("all recordings processed")


def main():
    # there are 2 grid cells in this recording and one of them (#7) looks theta modulated
    # recording_folder_path = '/mnt/datastore/Klara/Open_field_opto_tagging_p038/M13_2018-05-14_09-37-33_of/'
    # recording_folder_path = '/mnt/datastore/Klara/CA1_to_deep_MEC_in_vivo/M10_2021-12-10_08-37-27_of/'

    recording_list = []
    folder_path = "/mnt/datastore/Klara/CA1_to_deep_MEC_in_vivo/"
    recording_list.extend([f.path for f in os.scandir(folder_path) if f.is_dir()])
    process_recordings(recording_list)
    # recording_folder_path = '/mnt/datastore/Klara/CA1_to_deep_MEC_in_vivo_extras/PSAM/M10_2021-11-26_16-07-10_of'
    # analyse_waveform_shapes(recording_folder_path)


if __name__ == '__main__':
    main()