# import feather   # install feather-format not feather
import pyarrow.feather as feather
import glob
import os
from Helpers import open_ephys_IO
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, hilbert


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def sort_folder_names(list_of_names):
    list_of_names.sort(key=lambda x: int(x.split('CH')[1].split('.')[0]))
    return list_of_names


def bandpass_filter(data, low=5, high=9, fs=30000):
    if len(data.shape) == 1:
        filtered_data = butter_bandpass_filter(data, lowcut=low, highcut=high, fs=fs, order=2)

    else:
        filtered_data = np.zeros((data.shape[0], data.shape[1]))
        for channel in range(data.shape[0]):
            filtered_data[channel] = butter_bandpass_filter(data[channel, :], lowcut=low, highcut=high, fs=fs, order=2)
    return filtered_data


def load_all_channels(path, just_load_one=False):
    """
    Function to laod all channels in folder.
    """
    sorted_list_of_folders = sort_folder_names(glob.glob(path + '/*CH*continuous'))
    all_channels = False
    is_loaded = False
    is_first = True
    channel_count = 0
    for file_path in sorted_list_of_folders:
        if os.path.exists(file_path):
            channel_data = open_ephys_IO.get_data_continuous(file_path).astype(np.int16)
            if just_load_one:
                return np.array(channel_data), True
            if is_first:
                all_channels = np.zeros((len(list(glob.glob(path + '/*CH*continuous'))), channel_data.size), np.int16)
                is_first = False
                is_loaded = True
            all_channels[channel_count, :] = channel_data
            channel_count += 1
    return all_channels, is_loaded


def plot_results(channel_data, filtered_data, angle):
    # this is not executed but useful for testing and debugging so I will leave them here for now
    plt.cla()
    plt.plot(channel_data[200000:400000], color='grey', label='raw voltage')
    plt.plot(filtered_data[200000:400000], color='skyblue', label='theta filtered')
    plt.xlabel('Time (sampling points)')
    plt.ylabel('Voltage (uV)')
    plt.legend()
    plt.show()

    plt.cla()
    plt.plot(channel_data[200000:210000], color='grey', label='raw voltage')
    plt.plot(filtered_data[200000:210000], color='skyblue', label='theta filtered')
    plt.plot(angle[200000:210000] * 10, color='red', label='angle')
    plt.xlabel('Time (sampling points)')
    plt.ylabel('Voltage (uV)')
    plt.legend()
    plt.show()


def calculate_theta_phase_angle(channel_data, theta_low=5, theta_high=9):
    filtered_data = bandpass_filter(channel_data, low=theta_low, high=theta_high)  # filtered in theta range
    analytic_signal = hilbert(filtered_data)  # hilbert transform https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html
    angle = np.angle(analytic_signal)  # this is the theta angle (radians)
    return angle


def calculate_and_save_theta_phase_angles(recording_folder_path, theta_low=5, theta_high=9):
    print('I will calculate theta phase angles and save them.')
    all_channels, is_loaded = load_all_channels(recording_folder_path, just_load_one=False)
    for channel in range(all_channels.shape[0]):
        channel_data = all_channels[channel, :]
        angle = calculate_theta_phase_angle(channel_data, theta_low=theta_low, theta_high=theta_high)
        np.save(recording_folder_path + 'channel_angle_' + str(channel) + '.npy', angle)  # save array with angles


def up_sample_position_data(position_data, upsample_factor):
    position_data = position_data.reset_index()
    position_data.index = range(0, upsample_factor * len(position_data), upsample_factor)
    position_data_with_nans = position_data.reindex(index=range(upsample_factor * len(position_data)))
    interpolated = position_data_with_nans.interpolate()
    # plt.plot(interpolated.position_x[:100], interpolated.position_y[:100])
    # plt.show()
    return interpolated


def add_down_sampled_angle_to_position_df(recording_folder_path, number_of_channels=16, upsample_factor=4, ephys_sampling=30000):
    print('Add downsampled angle to upsampled position data and save (position_theta.pkl)')
    position_df_path = recording_folder_path + 'MountainSort/DataFrames/position.pkl'
    if os.path.exists(position_df_path):
        position_data = pd.read_pickle(position_df_path)
        position_data_theta = up_sample_position_data(position_data, upsample_factor=upsample_factor)

    else:
        print('There is no position data for this recording: ' + recording_folder_path)
        return False
    for channel in range(number_of_channels):
        ch_angle = np.load(recording_folder_path + 'channel_angle_' + str(channel) + '.npy')
        ch_angle_downsampled = ch_angle[(position_data_theta.synced_time * ephys_sampling).astype(int)]
        position_data_theta['theta_angle_' + str(channel)] = ch_angle_downsampled

    position_data_theta.to_pickle(recording_folder_path + 'MountainSort/DataFrames/position_theta.pkl')


def get_theta_phase_for_spikes(spatial_firing, recording_folder_path):
    if not os.path.exists(recording_folder_path + 'channel_angle_0.npy'):
        print('Warning: you did not have the theta angles saved before trying to load them. I will try to generate '
              'them now.')
        # this function uses the theta angles saved as npy files
        calculate_and_save_theta_phase_angles(recording_folder_path, theta_low=5, theta_high=9)
    angles_at_spikes = []
    for index, cell in spatial_firing.iterrows():
        # load theta angle for primary channel (the ch where the cell had the highest amplitude
        primary_channel = ((cell.tetrode - 1) * 4 + cell.primary_channel) - 1  # numbering is from 1 in this df
        corresponding_theta_angle = np.load(recording_folder_path + 'channel_angle_' + str(primary_channel) + '.npy')
        phase_angles_at_spikes = corresponding_theta_angle[cell.firing_times.astype(int)]
        angles_at_spikes.append(phase_angles_at_spikes)
    return angles_at_spikes


def add_theta_phase_to_firing_data(recording_folder_path):
    """
    Loads theta phase angles and finds the angle that corresponds to each spike. This function currently saves a new
    spatial_firing data frame with the theta phase angle added to it.
    """
    print('Find theta phase for each action potential.')
    spatial_firing_path = recording_folder_path + 'MountainSort/DataFrames/spatial_firing.pkl'
    if os.path.exists(spatial_firing_path):
        spatial_firing = pd.read_pickle(spatial_firing_path)
        spatial_firing_theta = spatial_firing

    else:
        print('There is no spatial firing data for this recording: ' + recording_folder_path)
        return False

    angles_at_spikes = get_theta_phase_for_spikes(spatial_firing, recording_folder_path)
    spatial_firing_theta['theta_phase_angle'] = angles_at_spikes
    spatial_firing_theta.to_pickle(recording_folder_path + 'MountainSort/DataFrames/spatial_firing_theta.pkl')


def save_example_firing_data_for_cell(recording_folder_path, df_path, cluster_id):
    spatial_firing_theta = pd.read_pickle(recording_folder_path + df_path + '/spatial_firing_theta.pkl')
    data_for_example_cell = spatial_firing_theta[spatial_firing_theta.cluster_id == cluster_id]
    cell_data_frame = pd.DataFrame()
    cell_data_frame['firing_times'] = data_for_example_cell.firing_times.values[0].astype(int)
    cell_data_frame['position_x'] = data_for_example_cell.position_x.values[0]
    cell_data_frame['position_y'] = data_for_example_cell.position_y.values[0]
    cell_data_frame['theta_angle'] = data_for_example_cell.theta_phase_angle.values[0]
    # cell_data_frame.to_feather(recording_folder_path + df_path + '/spatial_firing_theta_cluster_' + str(cluster_id) + '.feather')
    feather.write_feather(cell_data_frame, recording_folder_path + df_path + '/spatial_firing_theta_cluster_' + str(cluster_id) + '.feather')
    return data_for_example_cell


def save_example_position_data_for_cell(data_for_example_cell, recording_folder_path, df_path, cluster_id):
    cell = data_for_example_cell.iloc[0]
    position_theta = pd.read_pickle(recording_folder_path + df_path + '/position_theta.pkl')
    primary_channel = ((cell.tetrode - 1) * 4 + cell.primary_channel) - 1  # numbering is from 1 in this df
    theta_channel_name = "theta_angle_" + str(primary_channel)
    position_to_save = position_theta[["synced_time", "position_x", "position_y", "hd", "speed", theta_channel_name]]
    # position_to_save.to_feather(recording_folder_path + df_path + '/position_theta_cluster_' + str(cluster_id) + '.feather')
    feather.write_feather(position_to_save, recording_folder_path + df_path + '/position_theta_cluster_' + str(cluster_id) + '.feather')


def save_data_for_example_cell(recording_folder_path, cluster_id=7, df_path='MountainSort/DataFrames'):
    """
    Save data for an example cell as feather files. (These are easy to open in R).
    """
    print('Saving data for analysis in R for this cell: ' + recording_folder_path + ' cluster: ' + str(cluster_id))
    data_for_example_cell = save_example_firing_data_for_cell(recording_folder_path, df_path, cluster_id)
    save_example_position_data_for_cell(data_for_example_cell, recording_folder_path, df_path, cluster_id)


def plot_theta_phase_angle_histogram(recording_folder_path, phase_angles_at_spikes, cluster_id, fig_path='MountainSort/Figures/theta'):
    if not os.path.exists(recording_folder_path + fig_path):
        os.mkdir(recording_folder_path + fig_path)
    plt.cla()
    ax = plt.subplot(111)
    ax.hist(phase_angles_at_spikes, color='skyblue', bins=100)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('Theta phase', fontsize=16)
    plt.ylabel('Number of spikes', fontsize=16)
    plt.title('Cluster ' + str(cluster_id), fontsize=16)
    plt.savefig(recording_folder_path + fig_path + '/theta_phase_hist' + str(cluster_id) + '.png')
    plt.close()


def visualize_theta_modulation_for_cells(recording_folder_path):
    path = recording_folder_path + 'MountainSort/DataFrames/spatial_firing_theta.pkl'
    print('Plot histogram of spikes at differne theta angles. This is saved here: ' + path)
    if os.path.exists(path):
        spatial_firing = pd.read_pickle(path)
        for index, cell in spatial_firing.iterrows():
            phase_angles_at_spikes = cell.theta_phase_angle
            plot_theta_phase_angle_histogram(recording_folder_path, phase_angles_at_spikes, cell.cluster_id)


def analyse_theta_modulation(recording_folder_path):
    calculate_and_save_theta_phase_angles(recording_folder_path, theta_low=5, theta_high=9)
    add_down_sampled_angle_to_position_df(recording_folder_path, upsample_factor=4)  # upsample position data (120 Hz)
    add_theta_phase_to_firing_data(recording_folder_path)  # find theta phase for each spike
    # make separate df for example cell and save for R as a feather file
    save_data_for_example_cell(recording_folder_path, cluster_id=7)
    visualize_theta_modulation_for_cells(recording_folder_path)


def main():
    # there are 2 grid cells in this recording and one of them (#7) looks theta modulated
    recording_folder_path = '/mnt/datastore/Klara/Open_field_opto_tagging_p038/M13_2018-05-14_09-37-33_of/'
    analyse_theta_modulation(recording_folder_path)


if __name__ == '__main__':
    main()