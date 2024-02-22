import matplotlib.pylab as plt
from matplotlib  import cm
import numpy as np
from Helpers import plot_utility


def plot_avg_firing_combined(superficial, deep, path, name):
    fr_fig = plt.figure()
    ax = fr_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    fr_fig, ax = plot_utility.style_plot(ax)
    # ax.hist(all_cells.avgFR, bins=400, cumulative=True, histtype='step', normed=True, color='k')
    ax.hist(superficial.avgFR, bins=800, cumulative=True, histtype='step', normed=True, color='red')
    ax.hist(deep.avgFR, bins=800, cumulative=True, histtype='step', normed=True, color='navy')
    plt.xlim(0, 55)
    plt.ylim(0, 1)
    plt.xlabel('Average firing rate')
    plt.ylabel('Fraction')
    plt.savefig(path + 'avg_firing_rate_histogram_combined' + name + '.png')
    plt.close()


def plot_avg_firing_combined_excitatory(superficial, deep, path, name):
    fr_fig = plt.figure()
    ax = fr_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    fr_fig, ax = plot_utility.style_plot(ax)
    excitatory_superficial = superficial['avgFR'] <= 10
    excitatory_deep = deep['avgFR'] <= 10
    # ax.hist(all_cells.avgFR, bins=400, cumulative=True, histtype='step', normed=True, color='k')
    ax.hist(superficial.avgFR[excitatory_superficial], bins=800, cumulative=True, histtype='step', normed=True, color='red')
    ax.hist(deep.avgFR[excitatory_deep], bins=800, cumulative=True, histtype='step', normed=True, color='navy')
    plt.xlim(0, 9.5)
    plt.ylim(0, 1)
    plt.xlabel('Average firing rate')
    plt.ylabel('Number of cells')
    plt.savefig(path + 'avg_firing_rate_histogram_combined_excitatory' + name + '.png')
    plt.close()


def plot_avg_firing_combined_inhibitory(superficial, deep, path, name):
    fr_fig = plt.figure()
    ax = fr_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    fr_fig, ax = plot_utility.style_plot(ax)
    inhibitory_superficial = superficial['avgFR'] > 10
    inhibitory_deep = deep['avgFR'] > 10
    # ax.hist(all_cells.avgFR, bins=400, cumulative=True, histtype='step', normed=True, color='k')
    ax.hist(superficial.avgFR[inhibitory_superficial], bins=800, cumulative=True, histtype='step', normed=True, color='red')
    ax.hist(deep.avgFR[inhibitory_deep], bins=800, cumulative=True, histtype='step', normed=True, color='navy')
    plt.xlim(10, 55)
    plt.ylim(0, 1)
    plt.xlabel('Average firing rate')
    plt.ylabel('Fraction')
    plt.savefig(path + 'avg_firing_rate_histogram_combined_excitatory' + name + '.png')
    plt.close()


def plot_avg_firing_combined_hist(superficial, deep, path, name):
    fr_fig = plt.figure()
    ax = fr_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    fr_fig, ax = plot_utility.style_plot(ax)
    # ax.hist(all_cells.avgFR, bins=400, cumulative=True, histtype='step', normed=True, color='k')
    ax.hist(superficial.avgFR, bins=50, histtype='step', color='red')
    ax.hist(deep.avgFR, bins=50, histtype='step', color='navy')
    plt.xlim(0, 55)
    #plt.ylim(0, 1)
    plt.xlabel('Average firing rate')
    plt.ylabel('Number of cells')
    plt.savefig(path + 'avg_firing_rate_histogram_combined' + name + '.png')
    plt.close()


def plot_firing_rate_hist(spike_data_frame, save_output_path, name):
    fr_fig = plt.figure()
    ax = fr_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    fr_fig, ax = plot_utility.style_plot(ax)
    ax.hist(spike_data_frame.avgFR[~spike_data_frame.avgFR.isnull()], bins=50, color='navy')
    plt.xlabel('Average firing rate')
    plt.ylabel('Fraction')
    plt.xlim(0, 100)
    plt.axvline(x=10, color='red')
    plt.savefig(save_output_path + 'avg_firing_rate_histogram' + name + '.png')
    plt.close()


def plot_max_fr_spatial(spike_data_frame, save_output_path):
    max_fr_spatial = plt.figure()
    ax = max_fr_spatial.add_subplot(111)
    max_fr_spatial, ax = plot_utility.style_plot(ax)
    bins = np.linspace(0, 200, 20)
    ax.hist(spike_data_frame.maxFRspatial, bins, color='navy')
    plt.xlabel('Maximum firing rate')
    plt.ylabel('Number of cells')
    plt.savefig(save_output_path + 'max_firing_rate_histogram_spatial.png')
    plt.close()


def plot_max_fr_spatial_excitatory(spike_data_frame, save_output_path):
    max_fr_spatial = plt.figure()
    ax = max_fr_spatial.add_subplot(111)
    max_fr_spatial, ax = plot_utility.style_plot(ax)
    bins = np.linspace(0, 20, 20)
    ax.hist(spike_data_frame.maxFRspatial, bins, color='navy')
    plt.xlabel('Maximum firing rate')
    plt.ylabel('Number of cells')
    plt.savefig(save_output_path + 'excitatory_max_firing_rate_histogram_spatial.png')
    plt.close()


def plot_max_fr_head_dir(spike_data_frame, save_output_path):
    max_fr_hd = plt.figure()
    ax = max_fr_hd.add_subplot(111)
    max_fr_hd, ax = plot_utility.style_plot(ax)
    bins = np.linspace(0, 200, 20)
    ax.hist(spike_data_frame.HD_maxFR, bins, color='navy')
    plt.xlabel('Maximum firing rate')
    plt.ylabel('Number of cells')
    plt.savefig(save_output_path + 'max_firing_rate_histogram_head_dir.png')
    plt.close()


def plot_max_fr_head_dir_excitatory(spike_data_frame, save_output_path):
    max_fr_hd = plt.figure()
    ax = max_fr_hd.add_subplot(111)
    max_fr_hd, ax = plot_utility.style_plot(ax)
    bins = np.linspace(0, 20, 20)
    ax.hist(spike_data_frame.HD_maxFR, bins, color='navy')
    plt.xlabel('Maximum firing rate')
    plt.ylabel('Number of cells')
    plt.savefig(save_output_path + 'excitatory_max_firing_rate_histogram_head_dir.png')
    plt.close()


def plot_grid_score_hist(spike_data_frame, save_output_path, name):
    grid_sc_fig = plt.figure()
    ax = grid_sc_fig.add_subplot(1, 1, 1)
    grid_sc_fig, ax = plot_utility.style_plot(ax)
    has_grid_score = spike_data_frame['gridscore'].notnull()
    ax.hist(spike_data_frame[has_grid_score].gridscore, color='navy', bins=50)
    plt.axvline(x=0, color='red')
    plt.xlim(-1.5, 1.5)
    plt.xlabel('Grid score', fontsize=22)
    plt.ylabel('Number of cells', fontsize=22)
    plt.savefig(save_output_path + 'grid_score_histogram' + name + '.png')
    plt.close()


def plot_grid_score_vs_firing_rate(spike_data_frame, save_output_path):
    grid_sc_vs_fr_fig = plt.figure()
    ax = grid_sc_vs_fr_fig.add_subplot(1, 1, 1)
    grid_sc_vs_fr_fig, ax = plot_utility.style_plot(ax)
    has_grid_score = spike_data_frame['gridscore'].notnull()
    x = spike_data_frame[has_grid_score].avgFR.values
    y = spike_data_frame[has_grid_score].gridscore.values
    plt.xlim(-1.5, 1.5)
    ax.plot(x, y, 'o', color='navy')
    plt.xlabel('Average firing rate')
    plt.ylabel('Grid score')
    plt.savefig(save_output_path + 'grid_score_vs_avgFR.png')
    plt.close()


def plot_hd_score_vs_firing_rate(spike_data_frame, save_output_path):
    hd_sc_vs_fr_fig = plt.figure()
    ax = hd_sc_vs_fr_fig.add_subplot(1, 1, 1)
    hd_sc_vs_fr_fig, ax = plot_utility.style_plot(ax)
    x = spike_data_frame.avgFR.values
    y = spike_data_frame.HD_maxFR.values
    ax.plot(x, y, 'o', color='navy')
    plt.xlabel('Average firing rate')
    plt.ylabel('Head-direction score')
    plt.savefig(save_output_path + 'hd_score_vs_avgFR.png')
    plt.close()


# this is to colour code scatter plots based on recording day
def add_colour_values(spike_data_frame):
    animals = spike_data_frame.animal.unique()
    colours = np.zeros(len(spike_data_frame))
    cell_counter = 0
    for animal in animals:
        rows_of_animal = spike_data_frame.animal == animal
        rows_of_animal_df = spike_data_frame[rows_of_animal]
        recording_days_animal = rows_of_animal_df.day.unique()
        total_number_of_days = len(recording_days_animal)
        for index, day in enumerate(recording_days_animal):
            day = spike_data_frame.day == day
            rows_of_animal_day = spike_data_frame[rows_of_animal & day]
            colours[cell_counter+index] = round((index + 1) / total_number_of_days * 100)
        cell_counter += index + 1
    spike_data_frame['colours'] = colours.astype(int)
    return spike_data_frame


def plot_grid_score_vs_hd_score(spike_data_frame, save_output_path, name):
    grid_sc_vs_hd_fig = plt.figure()
    ax = grid_sc_vs_hd_fig.add_subplot(1, 1, 1)
    grid_sc_vs_hd_fig, ax = plot_utility.style_plot(ax)
    has_grid_score = spike_data_frame['gridscore'].notnull()
    x = spike_data_frame[has_grid_score].gridscore.values
    y = spike_data_frame[has_grid_score].r_HD.values
    spike_data_frame = add_colour_values(spike_data_frame[has_grid_score])
    colour = spike_data_frame.colours.values
    plt.xlim(-1.5, 1.5)
    plt.axvline(x=0, color='red')
    ax.scatter(x, y, s=80, c=colour, marker='o', cmap=cm.binary)
    plt.xlabel('Grid score', fontsize=22)
    plt.ylabel('Head-direction score', fontsize=22)
    plt.savefig(save_output_path + 'grid_score_vs_hd_score' + name + '.png')
    plt.close()


def plot_spatial_coherence_vs_firing_rate(spike_data_frame, save_output_path):
    grid_spatial_coh_vs_fr_fig = plt.figure()
    ax = grid_spatial_coh_vs_fr_fig.add_subplot(1, 1, 1)
    grid_spatial_coh_vs_fr_fig, ax = plot_utility.style_plot(ax)
    x = spike_data_frame.avgFR.values
    y = spike_data_frame.spatialcoherence.values
    ax.plot(x, y, 'o', color='navy')
    plt.xlabel('Average firing rate')
    plt.ylabel('Spatial coherence')
    plt.savefig(save_output_path + 'spatial_coherence_vs_avgFR.png')
    plt.close()


def plot_all(spike_data_frame, save_output_path):
    plot_max_fr_spatial(spike_data_frame, save_output_path)
    plot_max_fr_spatial_excitatory(spike_data_frame, save_output_path)
    plot_max_fr_head_dir(spike_data_frame, save_output_path)
    plot_max_fr_head_dir_excitatory(spike_data_frame, save_output_path)

    plot_grid_score_hist(spike_data_frame, save_output_path, '_all_cells')
    spike_data_frame_l2 = spike_data_frame.loc[spike_data_frame['location'] == 2]
    spike_data_frame_l3 = spike_data_frame.loc[spike_data_frame['location'] == 3]
    spike_data_frame_l5 = spike_data_frame.loc[spike_data_frame['location'] == 5]
    spike_data_frame_superficial = spike_data_frame.loc[spike_data_frame['location'].isin([2, 3])]
    spike_data_frame_superficial_last_days = spike_data_frame_superficial.tail(4)
    spike_data_frame_l5_last_days = spike_data_frame_l5.tail(4)

    plot_grid_score_hist(spike_data_frame_l2, save_output_path, '_L2')
    plot_grid_score_hist(spike_data_frame_l3, save_output_path, '_L3')
    plot_grid_score_hist(spike_data_frame_l5, save_output_path, '_L5')
    plot_grid_score_hist(spike_data_frame_superficial, save_output_path, '_superficial')

    plot_grid_score_vs_hd_score(spike_data_frame, save_output_path, '_all_cells')
    plot_grid_score_vs_hd_score(spike_data_frame_l2, save_output_path, '_L2')
    plot_grid_score_vs_hd_score(spike_data_frame_l3, save_output_path, '_L3')
    plot_grid_score_vs_hd_score(spike_data_frame_l5, save_output_path, '_L5')
    plot_grid_score_vs_hd_score(spike_data_frame_superficial, save_output_path, '_superficial')

    plot_firing_rate_hist(spike_data_frame, save_output_path, '_all_cells')
    plot_firing_rate_hist(spike_data_frame_superficial, save_output_path, '_superficial')
    plot_firing_rate_hist(spike_data_frame_l5, save_output_path, '_L5')



    plot_avg_firing_combined(spike_data_frame_superficial, spike_data_frame_l5, save_output_path, '_combined')
    plot_avg_firing_combined_excitatory(spike_data_frame_superficial, spike_data_frame_l5, save_output_path, '_combined_excitatory')
    plot_avg_firing_combined_inhibitory(spike_data_frame_superficial, spike_data_frame_l5, save_output_path, '_combined_inhibitory')
    plot_avg_firing_combined_hist(spike_data_frame_superficial, spike_data_frame_l5, save_output_path, '_combined_hist')
    plot_avg_firing_combined(spike_data_frame_l5_last_days, spike_data_frame_superficial_last_days, save_output_path, '_last_days_combined')


    plot_grid_score_vs_firing_rate(spike_data_frame, save_output_path)
    plot_hd_score_vs_firing_rate(spike_data_frame, save_output_path)

    plot_spatial_coherence_vs_firing_rate(spike_data_frame, save_output_path)

