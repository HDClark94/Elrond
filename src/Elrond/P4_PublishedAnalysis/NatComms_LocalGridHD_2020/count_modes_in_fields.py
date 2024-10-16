import glob
import matplotlib.pylab as plt
from Helpers import math_utility, plot_utility
import numpy as np
import os
import Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings
import pandas as pd
from rpy2 import robjects as robj
from scipy.stats import circstd
import scipy.stats


local_path = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_local_path() + '/field_modes/'
server_path_mouse = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_server_path_mouse()
server_path_rat = Published_analysis.NatComms_LocalGridHD_2020.folder_path_settings.get_server_path_rat()


def load_field_data(output_path, server_path, spike_sorter):
    if os.path.exists(output_path):
        field_data = pd.read_pickle(output_path)
        return field_data
    field_data_combined = pd.DataFrame()
    for recording_folder in glob.glob(server_path + '*'):
        os.path.isdir(recording_folder)
        data_frame_path = recording_folder + spike_sorter + '/DataFrames/shuffled_fields.pkl'
        if os.path.exists(data_frame_path):
            print('I found a field data frame.')
            field_data = pd.read_pickle(data_frame_path)
            if 'shuffled_data' in field_data:
                field_data_to_combine = field_data[['session_id', 'cluster_id', 'field_id', 'indices_rate_map',
                                                    'spike_times', 'number_of_spikes_in_field', 'position_x_spikes',
                                                    'position_y_spikes', 'hd_in_field_spikes', 'hd_hist_spikes',
                                                    'times_session', 'time_spent_in_field', 'position_x_session',
                                                    'position_y_session', 'hd_in_field_session', 'hd_hist_session',
                                                    'hd_histogram_real_data', 'time_spent_in_bins',
                                                    'field_histograms_hz']].copy()
                field_data_to_combine['normalized_hd_hist'] = field_data.hd_hist_spikes / field_data.hd_hist_session
                if 'hd_score' in field_data:
                    field_data_to_combine['hd_score'] = field_data.hd_score
                if 'grid_score' in field_data:
                    field_data_to_combine['grid_score'] = field_data.grid_score

                field_data_combined = field_data_combined.append(field_data_to_combine)
                print(field_data_combined.head())
    field_data_combined.to_pickle(output_path)
    return field_data_combined


# select accepted fields based on list of fields that were correctly identified by field detector
def tag_accepted_fields_mouse(field_data, accepted_fields):
    unique_id = field_data.session_id + '_' + field_data.cluster_id.apply(str) + '_' + (field_data.field_id + 1).apply(str)
    field_data['unique_id'] = unique_id
    unique_id = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)
    accepted_fields['unique_id'] = unique_id
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


# select accepted fields based on list of fields that were correctly identified by field detector
def tag_accepted_fields_rat(field_data, accepted_fields):
    unique_id = field_data.session_id + '_' + field_data.cluster_id.apply(str) + '_' + (field_data.field_id + 1).apply(str)
    unique_cell_id = field_data.session_id + '_' + field_data.cluster_id.apply(str)
    field_data['unique_id'] = unique_id
    field_data['unique_cell_id'] = unique_cell_id
    if 'Session ID' in accepted_fields:
        unique_id = accepted_fields['Session ID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)
    else:
        unique_id = accepted_fields['SessionID'] + '_' + accepted_fields['Cell'].apply(str) + '_' + accepted_fields['field'].apply(str)

    accepted_fields['unique_id'] = unique_id
    field_data['accepted_field'] = field_data.unique_id.isin(accepted_fields.unique_id)
    return field_data


# add cell type tp rat data frame
def add_cell_types_to_data_frame(field_data):
    cell_type = []
    for index, field in field_data.iterrows():
        if field.hd_score >= 0.5 and field.grid_score >= 0.4:
            cell_type.append('conjunctive')
        elif field.hd_score >= 0.5:
            cell_type.append('hd')
        elif field.grid_score >= 0.4:
            cell_type.append('grid')
        else:
            cell_type.append('na')

    field_data['cell type'] = cell_type

    return field_data


def resample_histogram(histogram):
    number_of_times_to_sample = robj.r(100000)
    seed = robj.r(210)
    hd_cluster_r = robj.FloatVector(histogram)
    rejection_sampling_r = robj.r['rejection.sampling']
    resampled_distribution = rejection_sampling_r(number_of_times_to_sample, hd_cluster_r, seed)
    return resampled_distribution


def fit_von_mises_mixed_model(resampled_distribution):
    fit_von_mises_mix = robj.r('vMFmixture')
    find_best_fit = robj.r('vMFmin')
    fit = find_best_fit(fit_von_mises_mix(resampled_distribution))
    print(fit)
    return fit


def get_model_fit_alpha_value(fit):
    get_model_fit_alpha = robj.r('get_model_fit_alpha')
    alpha = get_model_fit_alpha(fit)
    return alpha


def get_model_fit_theta_value(fit):
    get_model_fit_theta = robj.r('get_model_fit_theta')
    theta = get_model_fit_theta(fit)
    return theta


def get_estimated_density_function(fit):
    get_estimated_density = robj.r('get_estimated_density')
    estimated_density = get_estimated_density(fit)
    return estimated_density


def plot_modes_in_r(fit):
    plot_modes = robj.r("plot_modes")
    plot_modes(fit)


def generate_colors(number_of_firing_fields):
    colors = [[0, 1, 0], [1, 0.6, 0.3], [0, 1, 1], [1, 0, 1], [0.7, 0.3, 1], [0.6, 0.5, 0.4], [0.6, 0, 0]]  # green, orange, cyan, pink, purple, grey, dark red
    if number_of_firing_fields > len(colors):
        for i in range(number_of_firing_fields):
            colors.append(plot_utility.generate_new_color(colors, pastel_factor=0.9))
    return colors


def find_angles_and_lengths(theta):
    lengths = []
    angles = []
    number_of_modes = int(len(theta)/2)
    for mode in range(number_of_modes):
        length, angle = math_utility.cart2pol(np.asanyarray(theta)[mode][0], np.asanyarray(theta)[mode][1])
        lengths.append(length)
        angles.append(angle)
    return angles, lengths


def plot_modes_python(real_cell, hd_field_session, estimated_density, theta, field_id, path):
    hd_polar_fig = plt.figure()
    hd_polar_fig.set_size_inches(5, 5, forward=True)
    ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    theta_estimate = np.linspace(0, 2 * np.pi, 3601)  # x axis
    theta_real = np.linspace(0, 2 * np.pi, 361)  # x axis
    ax = plt.subplot(1, 1, 1, polar=True)
    ax = plot_utility.style_polar_plot(ax)
    angles, lengths = find_angles_and_lengths(theta)
    scale_for_lines = max(real_cell) / max(lengths)
    number_of_modes = int(len(theta) / 2)
    # for mode in range(number_of_modes):
        # ax.plot((0, angles[mode]), (0, lengths[mode]*scale_for_lines), color='red', linewidth=3)
    scale_for_density = max(real_cell) / max(estimated_density)
    colors = generate_colors(field_id + 1)
    # ax.plot(theta_estimate[:-1], estimated_density * scale_for_density, color=colors[field_id], alpha=0.3, linewidth=5)
    ax.plot(theta_real[:-1], list(real_cell), color=colors[field_id], linewidth=4)
    scale_for_occupancy = max(real_cell) / max(hd_field_session)
    ax.plot(theta_real[:-1], hd_field_session*scale_for_occupancy, color='gray', linewidth=2, alpha=0.6)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def get_mode_angles_degrees(theta):
    angles = []
    for mode in range(int(len(theta)/2)):
        length, angle = math_utility.cart2pol(np.asanyarray(theta)[mode][0], np.asanyarray(theta)[mode][1])
        angle *= 180 / np.pi
        angles.append(angle)
    return angles


def plot_modes_in_field(field, hd_histogram_field, fitted_density, theta):
    # plot_modes_in_r(fit)
    # concentration = np.asanyarray(theta)[0]
    path = local_path + 'estimated_modes/' + field.session_id + str(field.cluster_id) + str(field.field_id)
    if type(theta) == float:
        return
    else:
        plot_modes_python(hd_histogram_field, field.hd_hist_session, fitted_density, theta, field.field_id, path)


def analyze_histograms(field_data, output_path):
    if 'mode_angles' in field_data:
        return field_data
    robj.r.source('count_modes_circular_histogram.R')
    mode_angles = []
    fitted_densities = []
    thetas = []
    for index, field in field_data.iterrows():
        hd_histogram_field = field.normalized_hd_hist
        if np.isnan(hd_histogram_field).sum() > 0:
            print('skipping this field, it has nans')
            fitted_density = np.nan
            angles = np.nan
            theta = np.nan
        else:
            print('I will analyze ' + field.session_id)
            resampled_distribution = resample_histogram(hd_histogram_field)
            fit = fit_von_mises_mixed_model(resampled_distribution)
            # alpha = get_model_fit_alpha_value(fit)  # probability, the relative strength of belief in that mode
            theta = get_model_fit_theta_value(fit)
            angles = get_mode_angles_degrees(theta)
            fitted_density = get_estimated_density_function(fit)
        mode_angles.append(angles)
        fitted_densities.append(fitted_density)
        thetas.append(theta)
    field_data['fitted_density'] = fitted_densities
    field_data['mode_angles'] = mode_angles
    field_data['theta'] = thetas
    field_data.to_pickle(output_path)
    return field_data


def plot_fitted_field_results_with_occupancy(field_data):
    for index, field in field_data.iterrows():
        hd_histogram_field = field.normalized_hd_hist
        fitted_density = field.fitted_density
        theta = field.theta

        plot_modes_in_field(field, hd_histogram_field, fitted_density, theta)


def format_bar_chart(ax, x_label, y_label):
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.gcf().subplots_adjust(left=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel(x_label, fontsize=30)
    ax.set_ylabel(y_label, fontsize=30)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    return ax


def calculate_std_of_modes_for_cells(field_data, animal):
    # print(animal + ' modes are analyzed')
    # print('Plot standard deviation of modes for grid and conjunctive cells.')
    field_data['unique_cell_id'] = field_data.session_id + field_data.cluster_id.map(str)

    list_of_cells = np.unique(list(field_data.unique_cell_id))
    std_mode_angles_cells = []
    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        mode_angles = field_data.loc[field_data['unique_cell_id'] == cell_id].mode_angles
        # print(mode_angles)
        if mode_angles.notnull().sum() > 1:  # only do this for cells with multiple fields
            modes = []
            for field_mode in mode_angles:
                if type(field_mode) == float:
                    if ~np.isnan(field_mode):
                        modes.extend(field_mode)
                else:
                    modes.extend(field_mode)

            std_modes = circstd(modes, high=180, low=-180)
            std_mode_angles_cells.extend([std_modes] * len(field_data.loc[field_data['unique_cell_id'] == cell_id]))
        else:
            std_mode_angles_cells.extend([np.nan] * len(field_data.loc[field_data['unique_cell_id'] == cell_id]))
    field_data['angles_std_cell'] = std_mode_angles_cells


def get_mode_std_for_cell(field_data):
    std_cells = []
    list_of_cells = np.unique(list(field_data.unique_cell_id))
    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        std_angles = field_data.loc[field_data['unique_cell_id'] == cell_id].angles_std_cell
        if type(std_angles) is float:
            if ~np.isnan(std_angles):
                std_cells.extend(std_angles)
        else:
            std_cells.extend([np.asanyarray(std_angles)[0]])
    std_cells = [x for x in std_cells if ~np.isnan(x)]
    return std_cells


def plot_std_of_modes(field_data, animal):
    grid_cells = field_data['cell type'] == 'grid'
    conjunctive_cells = field_data['cell type'] == 'conjunctive'
    accepted_field = field_data['accepted_field'] == True
    grid_modes_std_cell = get_mode_std_for_cell(field_data[accepted_field & grid_cells])
    conjunctive_modes_std_cell = get_mode_std_for_cell(field_data[accepted_field & conjunctive_cells])
    fig, ax = plt.subplots()
    ax = format_bar_chart(ax, 'Standard dev of modes / cell', 'Proportion')
    plt.hist(grid_modes_std_cell, color='navy', weights=get_weights_normalized_hist(grid_modes_std_cell), bins=range(0, 180, 15), alpha=0.7)
    plt.hist(conjunctive_modes_std_cell, color='red', weights=get_weights_normalized_hist(conjunctive_modes_std_cell), bins=range(0, 180, 15), alpha=0.7)
    plt.savefig(local_path + animal + '_std_of_modes_of_grid_and_conj_cells')
    plt.close()

    fig, ax = plt.subplots()

    plt.yticks([0, 1])
    ax = format_bar_chart(ax, 'Standard dev of modes / cell', 'Cumulative probability')
    values, base = np.histogram(grid_modes_std_cell, bins=40)
    cumulative = np.cumsum(values / len(grid_modes_std_cell))
    plt.plot(base[:-1], cumulative, c='navy', linewidth=5)

    values, base = np.histogram(conjunctive_modes_std_cell, bins=40)
    cumulative = np.cumsum(values / len(conjunctive_modes_std_cell))
    plt.plot(base[:-1], cumulative, c='red', linewidth=5)

    plt.savefig(local_path + animal + '_std_of_modes_of_grid_and_conj_cells_cumulative')
    plt.close()


def get_number_of_modes_for_cell(field_data):
    number_of_modes_cells = []
    list_of_cells = np.unique(list(field_data.unique_cell_id))
    for cell in range(len(list_of_cells)):
        cell_id = list_of_cells[cell]
        number_of_modes = 0
        number_of_fields_with_modes = 0
        mode_angles = field_data.loc[field_data['unique_cell_id'] == cell_id].mode_angles
        if type(mode_angles) is float:
            if ~np.isnan(mode_angles):
                number_of_modes += 1
                number_of_fields_with_modes += 1
            else:
                continue
        else:
            for field in mode_angles:
                if type(field) is float:
                    if ~np.isnan(field):
                        number_of_modes += 1
                        number_of_fields_with_modes += 1
                else:
                    number_of_modes += len(field)
                    number_of_fields_with_modes += 1

        if number_of_fields_with_modes > 0:
            number_of_modes_cells.extend([number_of_modes / number_of_fields_with_modes])
    return number_of_modes_cells


def get_weights_normalized_hist(array_in):
    weights = np.ones_like(array_in) / float(len(array_in))
    return weights


def plot_histogram_of_number_of_modes(field_data, animal):
    grid_cells = field_data['cell type'] == 'grid'
    conjunctive_cells = field_data['cell type'] == 'conjunctive'
    accepted_field = field_data['accepted_field'] == True
    grid_number_of_modes_cell = get_number_of_modes_for_cell(field_data[accepted_field & grid_cells])
    conjunctive_number_of_modes_cell = get_number_of_modes_for_cell(field_data[accepted_field & conjunctive_cells])
    fig, ax = plt.subplots()
    ax = format_bar_chart(ax, 'Number of modes / cell', 'Proportion')
    plt.hist(grid_number_of_modes_cell, color='navy', weights=get_weights_normalized_hist(grid_number_of_modes_cell), alpha=0.7)
    plt.hist(conjunctive_number_of_modes_cell, color='red', weights=get_weights_normalized_hist(conjunctive_number_of_modes_cell), alpha=0.7)
    plt.savefig(local_path + animal + '_number_of_modes_per_field_grid_and_conj_cells')
    plt.close()

    fig, ax = plt.subplots()

    plt.yticks([0, 1])
    ax = format_bar_chart(ax, 'Number of modes / cell', 'Cumulative probability')
    values, base = np.histogram(grid_number_of_modes_cell, bins=40)
    cumulative = np.cumsum(values / len(grid_number_of_modes_cell))
    plt.plot(base[:-1], cumulative, c='navy', linewidth=5)

    values, base = np.histogram(conjunctive_number_of_modes_cell, bins=40)
    cumulative = np.cumsum(values / len(conjunctive_number_of_modes_cell))
    plt.plot(base[:-1], cumulative, c='red', linewidth=5)

    plt.savefig(local_path + animal + '_number_of_modes_per_field_grid_and_conj_cells_cumulative')
    plt.close()

    print(animal)
    print('Number of modes per grid field: ' + str(np.mean(grid_number_of_modes_cell)) + ' +/-' + str(np.std(grid_number_of_modes_cell)))


def compare_mode_distributions_of_grid_and_conj_cells(field_data, animal):
    grid_cells = field_data['cell type'] == 'grid'
    conjunctive_cells = field_data['cell type'] == 'conjunctive'
    accepted_field = field_data['accepted_field'] == True
    grid_modes_std = field_data[accepted_field & grid_cells].angles_std_cell.dropna()
    grid_modes_std = get_mode_std_for_cell(field_data[accepted_field & grid_cells])
    conjunctive_modes_std = field_data[accepted_field & conjunctive_cells].angles_std_cell.dropna()
    conjunctive_modes_std = get_mode_std_for_cell(field_data[accepted_field & conjunctive_cells])

    stat, p = scipy.stats.mannwhitneyu(grid_modes_std, conjunctive_modes_std)
    print('p value from mann-whitney test for grid and conj cells from ' + animal + ':')
    print(p)
    print('number of grid cells:' + str(len(grid_modes_std)))
    print('number of conjunctive cells:' + str(len(conjunctive_modes_std)))

    w, p_equal_var = scipy.stats.levene(grid_modes_std, conjunctive_modes_std)
    print('p value for equal var test (Levene): ' + str(p_equal_var))
    if p_equal_var <= 0.05:
        t, p_t = scipy.stats.ttest_ind(grid_modes_std, conjunctive_modes_std, axis=0, equal_var=True)
        print('T-test with equal variances result: ' + str(p_t))
    else:
        t, p_t = scipy.stats.ttest_ind(grid_modes_std, conjunctive_modes_std, axis=0, equal_var=False)
        print('T-test with unequal variances result: ' + str(p_t))


def process_circular_data(animal):
    # print('I am loading the data frame that has the fields')
    if animal == 'mouse':
        mouse_path = local_path + 'field_data_modes_mouse.pkl'
        field_data = load_field_data(mouse_path, server_path_mouse, '/MountainSort')
        field_data = analyze_histograms(field_data, mouse_path)
        accepted_fields = pd.read_excel(local_path + 'list_of_accepted_fields.xlsx')
        field_data = tag_accepted_fields_mouse(field_data, accepted_fields)
        field_data = add_cell_types_to_data_frame(field_data)
        calculate_std_of_modes_for_cells(field_data, 'mouse')
        plot_std_of_modes(field_data, 'mouse')
        plot_histogram_of_number_of_modes(field_data, 'mouse')
        compare_mode_distributions_of_grid_and_conj_cells(field_data, 'mouse')
        plot_fitted_field_results_with_occupancy(field_data)

    if animal == 'rat':
        rat_path = local_path + 'field_data_modes_rat.pkl'
        field_data = load_field_data(local_path + 'field_data_modes_rat.pkl', server_path_rat, '')
        accepted_fields = pd.read_excel(local_path + 'included_fields_detector2_sargolini.xlsx')
        field_data = tag_accepted_fields_rat(field_data, accepted_fields)
        field_data = analyze_histograms(field_data, rat_path)
        field_data = add_cell_types_to_data_frame(field_data)
        calculate_std_of_modes_for_cells(field_data, 'rat')
        plot_std_of_modes(field_data, 'rat')
        plot_histogram_of_number_of_modes(field_data, 'rat')
        compare_mode_distributions_of_grid_and_conj_cells(field_data, 'rat')
        plot_fitted_field_results_with_occupancy(field_data)


def main():
    process_circular_data('rat')
    process_circular_data('mouse')


if __name__ == '__main__':
    main()