import cmocean
import matplotlib.pylab as plt
import matplotlib.image as mpimg
import os
from Elrond.Helpers import plot_utility
import math
import numpy as np
import pandas as pd
import Elrond.settings as settings

from Elrond.P2_PostProcess.OpenField.Scores.head_direction import get_hd_histogram

def plot_position(position_data):
    plt.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=5)
    plt.close()

def plot_spikes_on_trajectory(spike_data, position_data, output_path):
    print('I will make scatter plots of spikes on the trajectory of the animal.')
    save_path = output_path + '/Figures/firing_scatters'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    raw_trajectory = plt.figure()
    raw_trajectory.set_size_inches(5, 5, forward=True)
    ax = raw_trajectory.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=2, zorder=1, alpha=0.7)
    plt.title('Trajectory', y=1.08, fontsize=24)
    plt.savefig(save_path + '/trajectory.png',dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    for cluster_index, cluster_id in enumerate(spike_data.cluster_id):
        cluster_df = spike_data[(spike_data.cluster_id == cluster_id)] # dataframe for that cluster

        spikes_on_track = plt.figure()
        spikes_on_track.set_size_inches(5, 5, forward=True)
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        ax.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=2, zorder=1, alpha=0.7)
        ax.scatter(cluster_df['position_x'].iloc[0], cluster_df['position_y'].iloc[0], color='red', marker='o', s=10, zorder=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            right=False,
            left=False,
            labelleft=False,
            labelbottom=False)  # labels along the bottom edge are off
        ax.set_aspect('equal')
        plt.title('Spikes on trajectory', y=1.08, fontsize=24)
        plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '_spikes_on_trajectory.png', dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.savefig(save_path + '/' + spike_data.session_id[cluster_id] + '_' + str(cluster_id + 1) + '_spikes_on_trajectory.pdf', bbox_inches='tight')
        plt.close()

def plot_coverage(position_heat_map, output_path):
    print('I will plot a heat map of the position of the animal to show coverage.')
    save_path = output_path + '/session'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    coverage = plt.figure()
    coverage.set_size_inches(5, 5, forward=True)
    ax = coverage.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    ax = plot_utility.style_open_field_plot(ax)
    position_heat_map = np.rot90(position_heat_map)
    coverage_fig = ax.imshow(position_heat_map, cmap=cmocean.cm.thermal, interpolation='nearest')
    coverage.colorbar(coverage_fig)
    plt.title('Coverage', y=1.08, fontsize=24)
    plt.savefig(save_path + '/heatmap.png', dpi=300)
    # plt.savefig(save_path + '/heatmap.pdf')
    plt.close()


def plot_firing_rate_vs_speed(spatial_firing, spatial_data, output_path):
    sampling_rate = 30
    print('I will plot spikes vs speed for the whole session excluding opto tagging.')
    save_path = output_path + '/Figures/firing_properties'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    speed = spatial_data.speed[~np.isnan(spatial_data.speed)]
    number_of_bins = math.ceil(max(speed)) - math.floor(min(speed))
    session_hist, bins_s = np.histogram(speed, bins=number_of_bins, range=(math.floor(min(speed)), math.ceil(max(speed))))

    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):

        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        speed_cluster = cluster_df['speed'].iloc[0]
        speed_cluster = sorted(speed_cluster)
        spike_hist = plt.figure()
        spike_hist.set_size_inches(5, 5, forward=True)
        ax = spike_hist.add_subplot(1, 1, 1)
        speed_hist, ax = plot_utility.style_plot(ax)
        if number_of_bins > 0:
            hist, bins = np.histogram(speed_cluster[1:], bins=number_of_bins, range=(math.floor(min(speed)), math.ceil(max(speed))))
            width = bins[1] - bins[0]
            center_bin = (bins[:-1] + bins[1:]) / 2
            center = center_bin[tuple([np.where(session_hist > sum(session_hist)*0.005)])]
            hist = np.array(hist, dtype=float)
            session_hist = np.array(session_hist, dtype=float)
            rate = np.divide(hist, session_hist, out=np.zeros_like(hist), where=session_hist != 0)
            rate = rate[tuple([np.where(session_hist[~np.isnan(session_hist)] > sum(session_hist)*0.005)])]
            plt.bar(center[0], rate[0]*sampling_rate, align='center', width=width, color='black')
        plt.xlabel('speed [cm/s]')
        plt.ylabel('firing rate [Hz]')
        plt.xlim(0, 30)
        plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '_speed_histogram.png', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()



def plot_firing_rate_maps(spatial_firing, output_path):
    print('I will make rate map plots.')
    save_path = output_path + '/Figures/rate_maps'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    rate_maps = []
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        firing_rate_map_original = cluster_df['firing_maps'].iloc[0]
        occupancy_map = cluster_df['occupancy_maps'].iloc[0]
        firing_rate_map_original[occupancy_map==0] = np.nan
        firing_rate_map = np.rot90(firing_rate_map_original)
        firing_rate_map_fig = plt.figure()
        firing_rate_map_fig.set_size_inches(5, 5, forward=True)
        ax = firing_rate_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax = plot_utility.style_open_field_plot(ax)
        cmap = plt.get_cmap('jet')
        cmap.set_bad("white")
        rate_map_img = ax.imshow(firing_rate_map, cmap=cmap, interpolation='nearest')
        firing_rate_map_fig.colorbar(rate_map_img)
        rate_maps.append(firing_rate_map)
        #plt.title('Firing rate map \n max fr: ' + str(round(cluster_df['max_firing_rate'].iloc[0], 2)) +
        #          ' Hz \n HS r: ' + str(round(cluster_df['rate_map_correlation_first_vs_second_half'].iloc[0], 2)) +
        #          ', % bins: ' + str(100-round(cluster_df['percent_excluded_bins_rate_map_correlation_first_vs_second_half_p'].iloc[0], 2)), y=1.08, fontsize=15)
        plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_rate_map_' + str(cluster_id) + '.png', dpi=300)
        # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_rate_map_' + str(cluster + 1) + '.pdf')
        plt.close()


    nrows = int(np.ceil(np.sqrt(len(spatial_firing))))
    ncols = nrows; i=0; j=0;
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(20, 20), squeeze=False)
    for rate_map in rate_maps:
        ax[j, i].imshow(rate_map, cmap=cmap, interpolation='nearest')
        i+=1
        if i==ncols:
            i=0; j+=1
    for j in range(nrows):
        for i in range(ncols):
            ax[j, i].spines['top'].set_visible(False)
            ax[j, i].spines['right'].set_visible(False)
            ax[j, i].spines['bottom'].set_visible(False)
            ax[j, i].spines['left'].set_visible(False)
            ax[j, i].set_xticks([])
            ax[j, i].set_yticks([])
            ax[j, i].xaxis.set_tick_params(labelbottom=False)
            ax[j, i].yaxis.set_tick_params(labelleft=False)
    plt.subplots_adjust(hspace=.1, wspace=.1, bottom=None, left=None, right=None, top=None)
    plt.savefig(save_path + '/all_firing_rates.png', dpi=400)
    plt.close()



def plot_hd(spatial_firing, position_data, prm):
    print('I will plot HD on open field maps as a scatter plot for each cluster.')
    save_path = prm.get_output_path() + '/Figures/head_direction_plots_2d'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        x_positions = cluster_df['position_x'].iloc[0]
        y_positions = cluster_df['position_y'].iloc[0]
        hd = cluster_df['hd'].iloc[0]
        hd_map_fig = plt.figure()
        hd_map_fig.set_size_inches(5, 5, forward=True)
        ax = hd_map_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax = plot_utility.style_open_field_plot(ax)
        ax.plot(position_data['position_x'], position_data['position_y'], color='black', linewidth=2, zorder=1,
                alpha=0.2)
        hd_plot = ax.scatter(x_positions, y_positions, s=20, c=hd, vmin=-180, vmax=180, marker='o', cmap=cmocean.cm.phase)
        plt.colorbar(hd_plot, fraction=0.046, pad=0.04)
        plt.title('Head direction at spikes', y=1.08, fontsize=24)
        plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_hd_map_' + str(cluster_id) + '.png', dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_hd_map_' + str(cluster + 1) + '.pdf', bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_polar_head_direction_histogram(spatial_firing, position_data, output_path):
    angles_whole_session = (np.array(position_data.hd) + 180) * np.pi / 180
    hd_hist = get_hd_histogram(angles_whole_session)
    hd_hist /= settings.sampling_rate

    print('I will make the polar HD plots now.')
    save_path = output_path + '/Figures/head_direction_plots_polar'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        hd_polar_fig = plt.figure()
        hd_polar_fig.set_size_inches(5, 5, forward=True)
        ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        hd_hist_cluster = cluster_df['hd_spike_histogram'].iloc[0]
        theta = np.linspace(0, 2*np.pi, 361)  # x axis
        ax = plt.subplot(1, 1, 1, polar=True)
        ax = plot_utility.style_polar_plot(ax)
        ax.plot(theta[:-1], hd_hist_cluster, color='red', linewidth=2)
        ax.plot(theta[:-1], hd_hist*(max(hd_hist_cluster)/max(hd_hist)), color='black', linewidth=2)
        plt.tight_layout()
        #  + '\nKuiper p: ' + str(spatial_firing.hd_p[cluster])
        plt.title('Head direction \n max fr: ' + str(round(cluster_df['max_firing_rate_hd'].iloc[0], 2)) + ' Hz' + ', hd score: ' + str(round(cluster_df['hd_score'].iloc[0], 2)) + '\n', y=1.08, fontsize=24)
        plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_hd_polar_' + str(cluster_id) + '.png', dpi=300, bbox_inches="tight")
        # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_hd_polar_' + str(cluster + 1) + '.pdf', bbox_inches="tight")
        plt.close()


# plot polar hd histograms without needing the whole df as an input
def plot_polar_hd_hist(hist_1, hist_2, cluster, save_path, color1='lime', color2='navy', title=''):
    hd_polar_fig = plt.figure()
    hd_polar_fig.set_size_inches(5, 5, forward=True)
    ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    ax = plt.subplot(1, 1, 1, polar=True)
    ax = plot_utility.style_polar_plot(ax)
    ax.plot(theta[:-1], hist_1, color=color1, linewidth=2)
    ax.plot(theta[:-1], hist_2, color=color2, linewidth=2)
    plt.title(title)
    # ax.plot(theta[:-1], hist_2 * (max(hist_1) / max(hist_2)), color='navy', linewidth=2)
    plt.tight_layout()
    plt.savefig(save_path + '_hd_polar_' + str(cluster + 1) + '.png', dpi=300, bbox_inches="tight")
    # plt.savefig(save_path + '_hd_polar_' + str(cluster + 1) + '.pdf', bbox_inches="tight")
    plt.close()


# plot polar hd histograms without needing the whole df as an input
def plot_single_polar_hd_hist(hist_1, cluster, save_path, color1='lime', title=''):
    hd_polar_fig = plt.figure()
    hd_polar_fig.set_size_inches(5, 5, forward=True)
    ax = hd_polar_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    ax = plt.subplot(1, 1, 1, polar=True)
    ax = plot_utility.style_polar_plot(ax)
    ax.plot(theta[:-1], hist_1, color=color1, linewidth=10)
    plt.title(title)
    # ax.plot(theta[:-1], hist_2 * (max(hist_1) / max(hist_2)), color='navy', linewidth=2)
    plt.tight_layout()
    plt.savefig(save_path + '_hd_polar_' + str(cluster + 1) + '.png', dpi=300, bbox_inches="tight")
    # plt.savefig(save_path + '_hd_polar_' + str(cluster + 1) + '.pdf', bbox_inches="tight")
    plt.close()


def plot_rate_map_autocorrelogram(spatial_firing, output_path):
    print('I will make the rate map autocorrelogram grid plots now.')
    save_path = output_path + '/Figures/rate_map_autocorrelogram'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

        rate_map_autocorr_fig = plt.figure()
        rate_map_autocorr_fig.set_size_inches(5, 5, forward=True)
        ax = rate_map_autocorr_fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        rate_map_autocorr = cluster_df['rate_map_autocorrelogram'].iloc[0]
        if rate_map_autocorr.size:
            ax = plt.subplot(1, 1, 1)
            ax = plot_utility.style_open_field_plot(ax)
            autocorr_img = ax.imshow(rate_map_autocorr, cmap='jet', interpolation='nearest')
            rate_map_autocorr_fig.colorbar(autocorr_img)
            plt.tight_layout()
            plt.title('Autocorrelogram \n grid score: ' + str(round(cluster_df['grid_score'].iloc[0], 2)), fontsize=24)
            plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_rate_map_autocorrelogram_' + str(cluster_id) + '.png', dpi=300, bbox_inches="tight")
            # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_rate_map_autocorrelogram_' + str(cluster + 1) + '.pdf', bbox_inches="tight")
        plt.close()


def mark_firing_field_with_scatter(field, plot, colors, field_id, rate_map):
    y_max = rate_map.shape[0] -1
    for bin in field:
        plot.scatter(bin[0], y_max - bin[1], color=colors[field_id], marker='o', s=25)
    return plot


# generate more random colors if necessary
def generate_colors(number_of_firing_fields):
    colors = [[0, 1, 0], [1, 0.6, 0.3], [0, 1, 1], [1, 0, 1], [0.7, 0.3, 1], [0.6, 0.5, 0.4], [0.6, 0, 0]]  # green, orange, cyan, pink, purple, grey, dark red
    if number_of_firing_fields > len(colors):
        for i in range(number_of_firing_fields):
            colors.append(plot_utility.generate_new_color(colors, pastel_factor=0.9))
    return colors


def save_field_polar_plot(save_path, hd_hist_session, hd_hist_cluster, cluster_id, spatial_firing, colors, field_id, name):
    cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

    field_polar = plt.figure()
    field_polar.set_size_inches(5, 5, forward=True)
    theta = np.linspace(0, 2*np.pi, 361)  # x axis
    hd_plot_field = field_polar.add_subplot(1, 1, 1, polar=True)
    hd_plot_field = plot_utility.style_polar_plot(hd_plot_field)

    hd_plot_field.plot(theta[:-1], hd_hist_session*(max(hd_hist_cluster)/max(hd_hist_session)), color='black', linewidth=2, alpha=0.9)
    hd_plot_field.plot(theta[:-1], hd_hist_cluster, color=colors[field_id], linewidth=2)
    plt.tight_layout()
    if 'field_max_firing_rate' in spatial_firing:
        field_max_firing_rate = str(round(cluster_df['field_max_firing_rate'].iloc[0][field_id], 2))
    else:
        field_max_firing_rate = '?'

    plt.title(str(cluster_df['number_of_spikes_in_fields'].iloc[0][field_id]) + ' spikes'
              + ' in ' + str(round(cluster_df['time_spent_in_fields_sampling_points'].iloc[0][field_id]/30, 2))
              +' seconds\n max fr: ' + field_max_firing_rate + 'Hz \n',
              y=1.08, fontsize=24)

    plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_cluster_' + str(cluster_id) + name + str(field_id + 1) + '.png', dpi=300, bbox_inches="tight")
    # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_cluster_' + str(cluster + 1) + name + str(field_id + 1) + '.pdf', bbox_inches="tight")
    plt.close()


def plot_hd_for_firing_fields(spatial_firing, spatial_data, prm):
    print('I will make the polar HD plots for individual firing fields now.')
    save_path = prm.get_output_path() + '/Figures/firing_field_plots'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    #for cluster in range(len(spatial_firing)):

    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

        if 'firing_fields' in cluster_df:
            number_of_firing_fields = len(cluster_df['firing_fields'].iloc[0])
            firing_rate_map = cluster_df['firing_maps'].iloc[0]
            if number_of_firing_fields > 0:
                plt.clf()
                of_figure = plt.figure()
                plt.title('HD in detected fields', fontsize=24)
                of_figure.set_size_inches(5, 5, forward=True)
                of_plot = of_figure.add_subplot(1, 1, 1)
                of_plot.axis('off')
                firing_rate_map_90 = np.rot90(firing_rate_map)
                of_plot.imshow(firing_rate_map_90)

                firing_fields_cluster = cluster_df['firing_fields'].iloc[0]
                colors = generate_colors(number_of_firing_fields)

                for field_id, field in enumerate(firing_fields_cluster):
                    of_plot = mark_firing_field_with_scatter(field, of_plot, colors, field_id, firing_rate_map_90)
                    hd_hist_session = cluster_df['firing_fields_hd_session'].iloc[0][field_id]
                    hd_hist_session = np.array(hd_hist_session) / prm.get_sampling_rate()
                    hd_hist_cluster = np.array(cluster_df['firing_fields_hd_cluster'].iloc[0][field_id])
                    hd_hist_cluster_normalized = np.divide(hd_hist_cluster, hd_hist_session, out=np.zeros_like(hd_hist_cluster), where=hd_hist_session != 0)

                    save_field_polar_plot(save_path, hd_hist_session, hd_hist_cluster_normalized, cluster_id, spatial_firing, colors, field_id, '_firing_field_')
                    save_field_polar_plot(save_path, hd_hist_session, hd_hist_cluster, cluster_id, spatial_firing, colors, field_id, '_firing_field_raw')

                plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_firing_fields_rate_map' + str(cluster_id) + '.png', dpi=300, bbox_inches="tight")
                # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_firing_fields_rate_map' + str(cluster + 1) + '.pdf', bbox_inches="tight")
                plt.close()


def plot_spikes_not_in_fields(spatial_firing, cluster_id, spatial_firing_cluster, of_plot):
    cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

    all_spikes_in_fields = np.hstack(np.array(cluster_df['spike_times_in_fields'].iloc[0]))
    mask_for_spikes_not_in_fields = ~np.in1d(cluster_df['firing_times'].iloc[0], all_spikes_in_fields)
    try:
        spike_times_not_in_fields = cluster_df['firing_times'].iloc[0][mask_for_spikes_not_in_fields]
    except:
        spike_times_not_in_fields = np.array(cluster_df['firing_times'].iloc[0])[mask_for_spikes_not_in_fields]
    not_in_fields_df = spatial_firing_cluster.loc[spatial_firing_cluster['firing_times'].isin(spike_times_not_in_fields)]
    of_plot.scatter(not_in_fields_df['x'].values, not_in_fields_df['y'].values, color='black', marker='o', s=6)


def make_df_for_cluster(spatial_firing, cluster_id):
    cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
    cluster_length = np.arange(len(cluster_df['firing_times'].iloc[0]))
    spatial_firing_cluster = pd.DataFrame(cluster_length)
    spatial_firing_cluster['x'] = cluster_df['position_x_pixels'].iloc[0]
    spatial_firing_cluster['y'] = cluster_df['position_y_pixels'].iloc[0]
    spatial_firing_cluster['hd'] = cluster_df['hd'].iloc[0]
    spatial_firing_cluster['firing_times'] = cluster_df['firing_times'].iloc[0]
    return spatial_firing_cluster

'''
Plot spikes on rate map colour coded to the [grid] field they belong to. This is only done for cells where fields
were detected.

'''


def plot_spikes_on_firing_fields(spatial_firing, prm):
    print('I will plot detected spikes colour coded in fields.')
    save_path = prm.get_output_path() + '/Figures/firing_fields_coloured_spikes'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster
        if 'firing_fields' in cluster_df:
            number_of_firing_fields = len(cluster_df['firing_fields'].iloc[0])
            if number_of_firing_fields > 0:
                plt.clf()
                of_figure = plt.figure()
                plt.title('spikes in fields')
                of_figure.set_size_inches(5, 5, forward=True)
                of_plot = of_figure.add_subplot(1, 1, 1)
                of_plot.axis('off')
                firing_fields_cluster = cluster_df['firing_fields'].iloc[0]
                colors = generate_colors(number_of_firing_fields)
                spatial_firing_cluster = make_df_for_cluster(spatial_firing, cluster_id)

                for field_id, field in enumerate(firing_fields_cluster):
                    spike_times_field = cluster_df['spike_times_in_fields'].iloc[0][field_id]
                    field_df = spatial_firing_cluster.loc[spatial_firing_cluster['firing_times'].isin(spike_times_field)]
                    of_plot.scatter(field_df['x'].values, field_df['y'].values, color=colors[field_id], marker='o', s=10)
                plot_spikes_not_in_fields(spatial_firing, cluster_id, spatial_firing_cluster, of_plot)

                plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_firing_fields_coloured_spikes' + str(cluster_id) + '.png', dpi=300, bbox_inches="tight")
                # plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_firing_fields_coloured_spikes' + str(cluster + 1) + '.pdf', bbox_inches="tight")
                plt.close()


def make_combined_figure(spatial_firing, output_path):
    print('I will make the combined images now.')
    save_path = output_path + '/Figures/combined'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.close('all')
    figures_path = output_path + '/Figures/'

    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
        cluster_df = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

        coverage_path = figures_path + 'session/heatmap.png'
        spike_scatter_path = figures_path + 'firing_scatters/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '_spikes_on_trajectory.png'
        rate_map_path = figures_path + 'rate_maps/' + cluster_df['session_id'].iloc[0] + '_rate_map_' + str(cluster_id) + '.png'
        head_direction_polar_path = figures_path + 'head_direction_plots_polar/' + cluster_df['session_id'].iloc[0] + '_hd_polar_' + str(cluster_id) + '.png'
        head_direction_map_path = figures_path + 'head_direction_plots_2d/' + cluster_df['session_id'].iloc[0] + '_hd_map_' + str(cluster_id) + '.png'
        firing_fields_rate_map_path = figures_path + 'firing_field_plots/' + cluster_df['session_id'].iloc[0] + '_firing_fields_rate_map' + str(cluster_id) + '.png'
        speed_histogram_path = figures_path + 'firing_properties/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '_speed_histogram.png'
        firing_field_path = figures_path + 'firing_field_plots/' + cluster_df['session_id'].iloc[0] + '_cluster_' + str(cluster_id) + '_firing_field_'
        rate_map_autocorrelogram_path = figures_path + 'rate_map_autocorrelogram/' + cluster_df['session_id'].iloc[0] + '_rate_map_autocorrelogram_' + str(cluster_id) + '.png'
        speed_vs_firing_rate_path = figures_path + 'firing_properties/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '_speed_vs_firing_rate.png'

        number_of_rows = math.ceil(1/6) + 2
        grid = plt.GridSpec(number_of_rows, 5, wspace=0.025, hspace=0.05)
        if os.path.exists(speed_vs_firing_rate_path):
            speed_vs_rate = mpimg.imread(speed_vs_firing_rate_path)
            speed_vs_rate_plot = plt.subplot(grid[0, 3])
            speed_vs_rate_plot.axis('off')
            speed_vs_rate_plot.imshow(speed_vs_rate)
        if os.path.exists(coverage_path):
            coverage = mpimg.imread(coverage_path)
            coverage_plot = plt.subplot(grid[0, 4])
            coverage_plot.axis('off')
            coverage_plot.imshow(coverage)
        if os.path.exists(spike_scatter_path):
            spike_scatter = mpimg.imread(spike_scatter_path)
            spike_scatter_plot = plt.subplot(grid[1, 0])
            spike_scatter_plot.axis('off')
            spike_scatter_plot.imshow(spike_scatter)
        if os.path.exists(rate_map_path):
            rate_map = mpimg.imread(rate_map_path)
            rate_map_plot = plt.subplot(grid[1, 1])
            rate_map_plot.axis('off')
            rate_map_plot.imshow(rate_map)
        if os.path.exists(rate_map_autocorrelogram_path):
            rate_map_autocorr = mpimg.imread(rate_map_autocorrelogram_path)
            rate_map_autocorr_plot = plt.subplot(grid[1, 2])
            rate_map_autocorr_plot.axis('off')
            rate_map_autocorr_plot.imshow(rate_map_autocorr)
        if os.path.exists(head_direction_polar_path):
            polar_hd = mpimg.imread(head_direction_polar_path)
            polar_hd_plot = plt.subplot(grid[1, 3])
            polar_hd_plot.axis('off')
            polar_hd_plot.imshow(polar_hd)
        if os.path.exists(head_direction_map_path):
            hd_map = mpimg.imread(head_direction_map_path)
            hd_map_plot = plt.subplot(grid[1, 4])
            hd_map_plot.axis('off')
            hd_map_plot.imshow(hd_map)
        if os.path.exists(firing_fields_rate_map_path):
            firing_fields = mpimg.imread(firing_fields_rate_map_path)
            firing_fields_plot = plt.subplot(grid[2, 0])
            firing_fields_plot.axis('off')
            firing_fields_plot.imshow(firing_fields)
        plt.savefig(save_path + '/' + cluster_df['session_id'].iloc[0] + '_' + str(cluster_id) + '.png', dpi=1000)
        plt.close()


