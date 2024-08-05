import matplotlib.pylab as plt
import numpy as np
import Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings
import Published_analysis.NatComms_LocalGridHD_2020.shuffle_field_analysis
import Published_analysis.NatComms_LocalGridHD_2020.false_positives
import pandas as pd
import PostSorting.parameters
from Helpers import plot_utility

import scipy
import scipy.stats


local_path = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path()
analysis_path = local_path + '/plot_hd_tuning_vs_shuffled_fields/'

prm = PostSorting.parameters.Parameters()
prm.set_pixel_ratio(440)
prm.set_sampling_rate(30000)


def add_cell_types_to_data_frame(spatial_firing):
    cell_type = []
    for index, cell in spatial_firing.iterrows():
        if cell.hd_score >= 0.5 and cell.grid_score >= 0.4:
            cell_type.append('conjunctive')
        elif cell.hd_score >= 0.5:
            cell_type.append('hd')
        elif cell.grid_score >= 0.4:
            cell_type.append('grid')
        else:
            cell_type.append('na')

    spatial_firing['cell type'] = cell_type

    return spatial_firing


def add_combined_id_to_df(spatial_firing):
    animal_ids = [session_id.split('_')[0] for session_id in spatial_firing.session_id.values]
    spatial_firing['animal'] = animal_ids

    dates = [session_id.split('_')[1] for session_id in spatial_firing.session_id.values]

    cluster = spatial_firing.cluster_id.values
    combined_ids = []
    for cell in range(len(spatial_firing)):
        id = animal_ids[cell] + '-' + dates[cell] + '-Cluster-' + str(cluster[cell])
        combined_ids.append(id)
    spatial_firing['false_positive_id'] = combined_ids
    return spatial_firing


# select accepted fields based on list of fields that were correctly identified by field detector
def tag_accepted_fields_mouse(field_data, accepted_fields):
    unique_id = field_data.session_id + '_' + field_data.cluster_id.apply(str) + '_' + (field_data.field_id + 1).apply(str)
    field_data['unique_id'] = unique_id
    unique_id = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)
    accepted_fields['unique_id'] = unique_id
    field_data['unique_cell_id'] = field_data.session_id + '_' + field_data.cluster_id.apply(str)
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


def plot_bar_chart_for_cells_percentile_error_bar(spatial_firing, path, animal, shuffle_type='occupancy', sampling_rate_video=30):
    counter = 0
    for index, cell in spatial_firing.iterrows():
        mean = np.append(cell['shuffled_means'], cell['shuffled_means'][0])
        percentile_95 = np.append(cell['error_bar_95'], cell['error_bar_95'][0])
        percentile_5 = np.append(cell['error_bar_5'], cell['error_bar_5'][0])
        field_spikes_hd = cell['hd_in_field_spikes']
        time_spent_in_bins = cell['time_spent_in_bins']
        # shuffled_histograms_hz = cell['field_histograms_hz']
        real_data_hz = np.histogram(field_spikes_hd, bins=20)[0] * sampling_rate_video / time_spent_in_bins
        max_rate = np.round(real_data_hz.max(), 2)
        x_pos = np.linspace(0, 2*np.pi, real_data_hz.shape[0] + 1)

        ax = plt.subplot(1, 1, 1, polar=True)
        ax = plot_utility.style_polar_plot(ax)
        x_labels = ["0", "", "", "", "", "90", "", "", "", "", "180", "", "", "", "", "270", "", "", "", ""]
        plt.xticks(x_pos, x_labels)
        ax.fill_between(x_pos, mean - percentile_5, percentile_95 + mean, color='grey', alpha=0.4)
        ax.plot(x_pos, mean, color='grey', linewidth=5, alpha=0.7)
        observed_data = np.append(real_data_hz, real_data_hz[0])
        ax.plot(x_pos, observed_data, color='navy', linewidth=5)
        plt.title('\n' + str(max_rate) + ' Hz', fontsize=20, y=1.08)
        plt.subplots_adjust(top=0.85)
        plt.savefig(analysis_path + animal + '_' + shuffle_type + '/' + str(counter) + str(cell['session_id']) + str(cell['cluster_id']) + '_percentile_polar_' + str(cell.percentiles_correction) + '.png')
        plt.close()
        counter += 1


def get_number_of_directional_fields(fields, tag='grid'):
    percentiles_no_correction = []
    percentiles_correction = []
    for index, field in fields.iterrows():
        percentile = scipy.stats.percentileofscore(field.number_of_different_bins_shuffled, field.number_of_different_bins)
        percentiles_no_correction.append(percentile)

        percentile = scipy.stats.percentileofscore(field.number_of_different_bins_shuffled_corrected_p, field.number_of_different_bins_bh)
        percentiles_correction.append(percentile)

    fields['percentiles_correction'] = percentiles_correction

    print(tag)
    print('Number of fields: ' + str(len(fields)))
    print('Number of directional fields [without correction]: ')
    print(np.sum(np.array(percentiles_no_correction) > 95))
    fields['directional_no_correction'] = np.array(percentiles_no_correction) > 95

    print('Number of directional fields [with BH correction]: ')
    print(np.sum(np.array(percentiles_correction) > 95))
    fields['directional_correction'] = np.array(percentiles_correction) > 95
    fields.to_pickle(analysis_path + tag + 'fields.pkl')
    return fields


def plot_hd_vs_shuffled():
    mouse_df_path = analysis_path + 'shuffled_field_data_all_mice.pkl'
    mouse_df = pd.read_pickle(mouse_df_path)
    all_cell_path = analysis_path + 'all_mice_df.pkl'
    all_cells = pd.read_pickle(all_cell_path)
    accepted_fields = pd.read_excel(analysis_path + 'list_of_accepted_fields.xlsx')
    df = tag_accepted_fields_mouse(mouse_df, accepted_fields)
    good_cells = df.accepted_field == True
    df_good_cells = df[good_cells]
    df = add_cell_types_to_data_frame(df_good_cells)
    grid_cells = df['cell type'] == 'grid'
    df_grid = df[grid_cells]

    df_grid = get_number_of_directional_fields(df_grid, tag='grid')
    print('mouse')
    df_grid = Published_analysis.NatComms_LocalGridHD_2020.shuffle_field_analysis.add_rate_map_values_to_field_df_session(all_cells, df_grid)
    df_grid = Published_analysis.NatComms_LocalGridHD_2020.shuffle_field_analysis.shuffle_field_data(df_grid, analysis_path, 20, number_of_times_to_shuffle=1000, shuffle_type='distributive')
    df_grid = Published_analysis.NatComms_LocalGridHD_2020.shuffle_field_analysis.add_mean_and_std_to_field_df(df_grid, 30, 20)
    df_grid = Published_analysis.NatComms_LocalGridHD_2020.shuffle_field_analysis.add_percentile_values_to_df(df_grid, 30, number_of_bins=20)
    plot_bar_chart_for_cells_percentile_error_bar(df_grid, analysis_path, 'mouse', shuffle_type='distributive')



def main():
    plot_hd_vs_shuffled()


if __name__ == '__main__':
    main()