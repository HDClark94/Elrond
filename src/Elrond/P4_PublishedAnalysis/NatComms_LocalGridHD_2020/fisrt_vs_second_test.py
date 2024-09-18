import glob
import os
import Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings
import pandas as pd
import PostSorting.open_field_head_direction
import PostSorting.open_field_make_plots
import PostSorting.compare_first_and_second_half
import PostSorting.parameters

prm = PostSorting.parameters.Parameters()
prm.set_sorter_name('MountainSort')

local_path = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/field_histogram_shapes/'
server_path_mouse = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_server_path_mouse()
server_path_rat = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_server_path_rat()


def load_field_data(output_path, server_path, spike_sorter):
    field_data_combined = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + '/DataFrames/shuffled_fields.pkl'
        spatial_firing_path = recording_folder + spike_sorter + '/DataFrames/spatial_firing.pkl'
        position_path = recording_folder + spike_sorter + '/DataFrames/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a field data frame.')
            field_data = pd.read_pickle(data_frame_path)
            spatial_firing = pd.read_pickle(spatial_firing_path)
            position = pd.read_pickle(position_path)
            prm.set_file_path(recording_folder)
            spatial_firing = PostSorting.compare_first_and_second_half.analyse_first_and_second_halves(position, spatial_firing, prm)
            if 'shuffled_data' in field_data:
                field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id',
                                                    'field_histograms_hz', 'indices_rate_map', 'hd_in_field_spikes',
                                                    'hd_in_field_session', 'spike_times', 'times_session']].copy()
                field_data_to_combine['normalized_hd_hist'] = field_data.hd_hist_spikes / field_data.hd_hist_session
                if 'hd_score' in field_data:
                    field_data_to_combine['hd_score'] = field_data.hd_score
                if 'grid_score' in field_data:
                    field_data_to_combine['grid_score'] = field_data.grid_score
                rate_maps = []
                length_recording = []
                for cluster in range(len(field_data.cluster_id)):
                    rate_map = spatial_firing[field_data.cluster_id.iloc[cluster] == spatial_firing.cluster_id].firing_maps
                    rate_maps.append(rate_map)
                    length_of_recording = position.synced_time.values[-1]
                    length_recording.append(length_of_recording)

                field_data_to_combine['rate_map'] = rate_maps
                field_data_to_combine['recording_length'] = length_recording
                field_data_combined = field_data_combined.append(field_data_to_combine)
                print(field_data_combined.head())
    field_data_combined.to_pickle(output_path)
    return field_data_combined


def main():
    animal = 'mouse'
    if animal == 'mouse':
        mouse_path = local_path + 'field_data_modes_mouse.pkl'
        field_data = load_field_data(mouse_path, server_path_mouse, '/MountainSort')


if __name__ == '__main__':
    main()
