import glob
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from Helpers import plot_utility
import PostSorting.make_plots
import PostSorting.speed
import Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings
import os
import Published_analysis.NatComms_LocalGridHD_2020.false_positives


local_path_mouse = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/speed/all_mice_df.pkl'
local_path_rat = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/speed/all_rats_df.pkl'
local_path_simulated = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/speed/all_simulated_df.pkl'
path_to_data = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/speed/'
save_output_path = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/speed/'

server_path_mouse = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_server_path_mouse()
server_path_rat = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_server_path_rat()
server_path_simulated = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_server_path_simulated()


def add_speed_score_to_spatial_firing(output_path, server_path, animal, video_sampling, ephys_sample, spike_sorter='', df_path='/DataFrames'):
    if os.path.exists(output_path):
        spatial_firing = pd.read_pickle(output_path)
        return spatial_firing
    spatial_firing_data = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + df_path + '/spatial_firing.pkl'
        position_data_path = recording_folder + spike_sorter + df_path + '/position.pkl'
        if os.path.exists(data_frame_path):
            print('I found a firing data frame.')
            spatial_firing = pd.read_pickle(data_frame_path)
            position_data = pd.read_pickle(position_data_path)
            if 'grid_score' in spatial_firing:
                if animal == 'rat':
                    spatial_firing = spatial_firing[['session_id', 'cell_id', 'cluster_id', 'firing_times',
                                                    'number_of_spikes', 'hd', 'speed', 'mean_firing_rate', 'grid_score', 'hd_score']].copy()
                if animal == 'mouse':
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'tetrode', 'firing_times',
                                                     'number_of_spikes', 'hd', 'speed', 'mean_firing_rate', 'grid_score', 'hd_score']].copy()
                if animal == 'simulated':
                    spatial_firing = spatial_firing[['session_id', 'cluster_id', 'firing_times',
                                                    'hd', 'hd_spike_histogram', 'speed', 'max_firing_rate_hd', 'grid_score', 'hd_score']].copy()

                spatial_firing = PostSorting.speed.calculate_speed_score(position_data, spatial_firing, gauss_sd=250, sampling_rate_conversion=ephys_sample)

                save_path = save_output_path + animal + 'grid_speed_scatter_plots/'
                if os.path.exists(save_path) is False:
                    os.makedirs(save_path)
                PostSorting.speed.plot_speed_vs_firing_rate_grid(position_data, spatial_firing, ephys_sample, video_sampling, save_path)
                spatial_firing_data = spatial_firing_data.append(spatial_firing)
    spatial_firing_data.to_pickle(output_path)
    return spatial_firing_data


def add_combined_id_to_df(df_all_mice):
    animal_ids = [session_id.split('_')[0] for session_id in df_all_mice.session_id.values]
    dates = [session_id.split('_')[1] for session_id in df_all_mice.session_id.values]
    tetrode = df_all_mice.tetrode.values
    cluster = df_all_mice.cluster_id.values

    combined_ids = []
    for cell in range(len(df_all_mice)):
        id = animal_ids[cell] + '-' + dates[cell] + '-Tetrode-' + str(tetrode[cell]) + '-Cluster-' + str(cluster[cell])
        combined_ids.append(id)
    df_all_mice['false_positive_id'] = combined_ids
    return df_all_mice


def tag_false_positives(all_cells, animal):
    if animal == 'mouse':
        false_positives_path = path_to_data + 'false_positives_all.txt'
        list_of_false_positives = Published_analysis.NatComms_LocalGridHD_2020.false_positives.get_list_of_false_positives(false_positives_path)
        all_cells = add_combined_id_to_df(all_cells)
        all_cells['false_positive'] = all_cells['false_positive_id'].isin(list_of_false_positives)
    else:
        all_cells['false_positive'] = np.full(len(all_cells), False)
    return all_cells


def add_cell_types_to_data_frame(cells):
    cell_type = []
    for index, field in cells.iterrows():
        if field.hd_score >= 0.5 and field.grid_score >= 0.4:
            cell_type.append('conjunctive')
        elif field.hd_score >= 0.5:
            cell_type.append('hd')
        elif field.grid_score >= 0.4:
            cell_type.append('grid')
        else:
            cell_type.append('na')

    cells['cell type'] = cell_type
    return cells


def plot_speed_dependence(spatial_firing, animal, tag, color='navy'):
    # plt.hist(spatial_firing.speed_score, alpha=0.5, normed=True, color='gray')
    plt.cla()
    fig, ax = plt.subplots()
    ax = plot_utility.format_bar_chart(ax, 'Speed score', 'Number of ' + tag + ' cells')
    plt.hist(spatial_firing.speed_score, alpha=0.8, color=color)
    plt.xlim(-1, 1)
    plt.savefig(save_output_path + animal + '_' + tag + '_cell_speed_scores.png')
    plt.close()


def process_data():
    spatial_firing = add_speed_score_to_spatial_firing(local_path_mouse, server_path_mouse, 'mouse', 30, 30000, spike_sorter='/MountainSort', df_path='/DataFrames')
    spatial_firing = tag_false_positives(spatial_firing, 'mouse')
    spatial_firing = add_cell_types_to_data_frame(spatial_firing)
    grid_cells = spatial_firing['cell type'] == 'grid'
    good_cell = spatial_firing.false_positive == False
    print('number of cells ' + str(len(spatial_firing[good_cell].speed_score)))
    median_speed_score = np.median(spatial_firing[good_cell].speed_score)
    print('[mouse] median speed score for all cells: ' + str(median_speed_score))
    print(np.std(spatial_firing[good_cell].speed_score))
    median_speed_score_grid = np.median(spatial_firing[grid_cells & good_cell].speed_score)
    print('number of grid cells ' + str(len(spatial_firing[good_cell & grid_cells].speed_score)))
    print('median speed score for grid cells: ' + str(median_speed_score_grid))
    print(np.std(spatial_firing[grid_cells & good_cell].speed_score))
    plot_speed_dependence(spatial_firing[grid_cells & good_cell], 'mouse', 'grid')
    plot_speed_dependence(spatial_firing[good_cell], 'mouse', 'all', color='gray')

    spatial_firing = add_speed_score_to_spatial_firing(local_path_rat, server_path_rat, 'rat', 50, 1, spike_sorter='', df_path='/DataFrames')
    spatial_firing['false_positive'] = False
    spatial_firing = add_cell_types_to_data_frame(spatial_firing)
    grid_cells = spatial_firing['cell type'] == 'grid'
    good_cell = spatial_firing.false_positive == False
    median_speed_score = np.median(spatial_firing[good_cell].speed_score)
    print('number of cells ' + str(len(spatial_firing[good_cell].speed_score)))
    print('[rat] median speed score for all cells: ' + str(median_speed_score))
    print(np.std(spatial_firing[good_cell].speed_score))
    median_speed_score_grid = np.median(spatial_firing[grid_cells & good_cell].speed_score)
    print('number of grid cells ' + str(len(spatial_firing[good_cell & grid_cells].speed_score)))
    print('median speed score for grid cells: ' + str(median_speed_score_grid))
    print(np.std(spatial_firing[grid_cells & good_cell].speed_score))
    plot_speed_dependence(spatial_firing[grid_cells & good_cell], 'rat', 'grid')
    plot_speed_dependence(spatial_firing[good_cell], 'rat', 'all', color='gray')


def main():
    process_data()


if __name__ == '__main__':
    main()