import glob
import numpy as np
import os
import Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings
import pandas as pd

local_path = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/data_for_modeling/density_ratio/'
local_path_fields = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/data_for_modeling/density_ratio_fields/'
false_positive_path = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/data_for_modeling/'
server_path_mouse = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_server_path_mouse()
server_path_rat = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_server_path_rat()


def save_trajectory_field_data_for_cell(position, spatial_firing, accepted_fields):
    for index, cluster in spatial_firing.iterrows():
        cluster_id = spatial_firing.cluster_id[index]
        session_id = spatial_firing.session_id[index]
        number_of_firing_fields = len(spatial_firing.firing_fields[index])
        if number_of_firing_fields > 0:
            hd = position.hd
            times_position = position.synced_time
            field_id = np.zeros(len(hd))
            # accepted_field_ids = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].astype(str) + '_' + accepted_fields['field'].astype(str)
            for field in range(len(cluster.firing_fields)):
                field_name = cluster.session_id + '_' + str(cluster.cluster_id) + '_' + str(field + 1)
                # if any(field_name in s for s in accepted_field_ids):
                occupancy_times = cluster.times_in_session_fields[field]
                mask_for_occupancy = np.in1d(times_position, occupancy_times)
                field_id[mask_for_occupancy] = field + 1
                #else:
                    #print('excluded field: ' + field_name)

            trajectory_df_to_save = pd.DataFrame()
            trajectory_df_to_save['hd'] = hd
            trajectory_df_to_save['field_id'] = field_id
            trajectory_df_to_save['cluster_id'] = cluster.cluster_id
            trajectory_df_to_save['session_id'] = cluster.session_id
            trajectory_df_to_save.to_csv(local_path + session_id + str(cluster_id) + '.csv')


def save_trajectory_field_data_for_cell_spikes(spatial_firing, accepted_fields):
    for index, cluster in spatial_firing.iterrows():
        cluster_id = spatial_firing.cluster_id[index]
        session_id = spatial_firing.session_id[index]
        number_of_firing_fields = len(spatial_firing.firing_fields[index])
        if number_of_firing_fields > 0:
            hd = spatial_firing.hd[cluster_id - 1]
            all_spike_times = spatial_firing.firing_times[cluster_id - 1]
            field_id = np.zeros(len(hd))
            for field in range(len(cluster.firing_fields)):
                # accepted_field_ids = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].astype(str) + '_' + accepted_fields['field'].astype(str)
                field_name = cluster.session_id + '_' + str(cluster.cluster_id) + '_' + str(field + 1)
                # if any(field_name in s for s in accepted_field_ids):
                spike_times_field = cluster.spike_times_in_fields[field]
                mask_for_occupancy = np.in1d(all_spike_times, spike_times_field)
                field_id[mask_for_occupancy] = int(field + 1)
                # else:
                    # print('excluded field: ' + field_name)

            spike_df_to_save = pd.DataFrame()
            spike_df_to_save['hd'] = hd
            spike_df_to_save['field_id'] = field_id
            spike_df_to_save['cluster_id'] = cluster.cluster_id
            spike_df_to_save['session_id'] = cluster.session_id
            spike_df_to_save.to_csv(local_path + session_id + str(cluster_id) + '_spikes.csv')


def load_data_frame_spatial_firing_modeling(output_path, server_path, spike_sorter='/MountainSort'):
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        firing_data_frame_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
        position_path = recording_folder + spike_sorter + '/DataFrames/position.pkl'
        accepted_fields = pd.read_excel(false_positive_path + 'list_of_accepted_fields.xlsx')
        if os.path.exists(firing_data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(firing_data_frame_path)
            position = pd.read_pickle(position_path)
            save_trajectory_field_data_for_cell_spikes(spatial_firing, accepted_fields)
            save_trajectory_field_data_for_cell(position, spatial_firing, accepted_fields)


def load_field_data_for_r(output_path, server_path):
    field_data_combined = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + '/MountainSort/DataFrames/shuffled_fields.pkl'
        if os.path.exists(data_frame_path):
            print('I found a field data frame.')
            field_data = pd.read_pickle(data_frame_path)

            if 'shuffled_data' in field_data:
                field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id', 'indices_rate_map',
                                                    'spike_times', 'number_of_spikes_in_field', 'position_x_spikes',
                                                    'position_y_spikes', 'hd_in_field_spikes', 'hd_hist_spikes',
                                                    'times_session', 'time_spent_in_field', 'position_x_session',
                                                    'position_y_session', 'hd_in_field_session', 'hd_hist_session',
                                                    'shuffled_means', 'shuffled_std',
                                                    'hd_histogram_real_data', 'time_spent_in_bins',
                                                    'field_histograms_hz',
                                                    'real_and_shuffled_data_differ_bin', 'number_of_different_bins',
                                                    'number_of_different_bins_shuffled', 'number_of_different_bins_bh',
                                                    'number_of_different_bins_holm',
                                                    'number_of_different_bins_shuffled_corrected_p']].copy()
                norm_hist = field_data.hd_hist_spikes / field_data.hd_hist_session
                normalized_hist = field_data.hd_hist_spikes / field_data.hd_hist_session

                save_combined = pd.DataFrame()
                for field in range(len(field_data)):
                    norm_hist[field][:norm_hist[field].size // 2] = normalized_hist[field][norm_hist[field].size // 2:]
                    norm_hist[field][norm_hist[0].size // 2:] = normalized_hist[field][:norm_hist[field].size // 2]

                    to_save = pd.DataFrame()
                    to_save['degrees'] = np.arange(-180, 180)
                    to_save['session_id'] = field_data_to_combine.session_id[field]
                    to_save['cluster_id'] = field_data_to_combine.cluster_id[field]
                    to_save['field_id'] = field_data_to_combine.field_id[field] + 1
                    to_save['rate_hist'] = norm_hist[field]
                    save_combined = save_combined.append(to_save)

                save_combined.to_csv(local_path_fields + to_save.session_id.iloc[0] + str(to_save.cluster_id.iloc[0]) + '.csv')


def process_data():
    load_field_data_for_r(local_path_fields, server_path_mouse)
    load_data_frame_spatial_firing_modeling(local_path, server_path_mouse, spike_sorter='/MountainSort')


def main():
    process_data()


if __name__ == '__main__':
    main()