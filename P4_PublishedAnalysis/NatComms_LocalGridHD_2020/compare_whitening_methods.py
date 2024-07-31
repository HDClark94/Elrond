import matplotlib.pylab as plt
import Published_analysis.NatComms_LocalGridHD_2020.false_positives
import pandas as pd
from Elrond.Helpers import plot_utility
import os


def plot_isolation_vs_noise_overlap(clusters_all, clusters_separate, params):
    save_output_path = params.get_save_output_path() + '/cluster_quality/'
    isolation = params.get_isolation()
    noise_overlap = params.get_noise_overlap()

    cluster_quality = plt.figure()
    ax = cluster_quality.add_subplot(1, 1, 1)
    cluster_quality, ax = plot_utility.style_plot(ax)
    x_all = clusters_all.isolation.values
    y_all = clusters_all.noiseoverlap.values
    x_separate = clusters_separate.isolation.values
    y_separate = clusters_separate.noiseoverlap.values
    ax.plot(x_separate, y_separate, 'o', markersize=3, color='red', label='Whitened separately')
    ax.plot(x_all, y_all, 'o', markersize=3, color='navy', label='Whitened together')

    plt.axvline(x=isolation, linewidth=3, color='black')
    plt.axhline(y=noise_overlap, linewidth=3, color='black')
    ax.legend(numpoints=1, frameon=False, loc='upper left')

    plt.xlabel('Isolation')
    plt.ylabel('Noise overlap')
    plt.savefig(save_output_path + 'isolation_vs_noise_overlap.png')


def plot_isolation_vs_snr(clusters_all, clusters_separate, params):
    save_output_path = params.get_save_output_path() + '/cluster_quality/'
    isolation = params.get_isolation()
    snr = params.get_snr()

    cluster_quality = plt.figure()
    ax = cluster_quality.add_subplot(1, 1, 1)
    cluster_quality, ax = plot_utility.style_plot(ax)
    x_all = clusters_all.isolation.values
    y_all = clusters_all.peakSNR.values
    x_separate = clusters_separate.isolation.values
    y_separate = clusters_separate.peakSNR.values
    ax.plot(x_separate, y_separate, 'o', markersize=3, color='red', label='Whitened separately')
    ax.plot(x_all, y_all, 'o', markersize=3, color='navy', label='Whitened together')

    plt.axvline(x=isolation, linewidth=3, color='black')
    plt.axhline(y=snr, linewidth=3, color='black')
    ax.legend(numpoints=1, frameon=False, loc='upper left')

    plt.xlabel('Isolation')
    plt.ylabel('Peak signal to noise ratio')
    plt.savefig(save_output_path + 'isolation_vs_snr.png')


def plot_noise_overlap_vs_snr(clusters_all, clusters_separate, params):
    save_output_path = params.get_save_output_path() + '/cluster_quality/'
    noise_overlap = params.get_noise_overlap()
    snr = params.get_snr()

    cluster_quality = plt.figure()
    ax = cluster_quality.add_subplot(1, 1, 1)
    cluster_quality, ax = plot_utility.style_plot(ax)
    x_all = clusters_all.noiseoverlap.values
    y_all = clusters_all.peakSNR.values
    x_separate = clusters_separate.noiseoverlap.values
    y_separate = clusters_separate.peakSNR.values
    ax.plot(x_separate, y_separate, 'o', markersize=3, color='red', label='Whitened separately')
    ax.plot(x_all, y_all, 'o', markersize=3, color='navy', label='Whitened together')

    plt.axvline(x=noise_overlap, linewidth=3, color='black')
    plt.axhline(y=snr, linewidth=3, color='black')
    ax.legend(numpoints=1, frameon=False)

    plt.xlabel('Noise overlap')
    plt.ylabel('Peak signal to noise ratio')
    plt.savefig(save_output_path + 'noise_overlap_vs_snr.png')


def plot_cluster_quality(clusters_all, clusters_separate, params):
    save_output_path = params.get_save_output_path() + '/cluster_quality/'
    if os.path.exists(save_output_path) is False:
        print('Cluster quality data be saved in {}.'.format(save_output_path))
        os.makedirs(save_output_path)

    plot_noise_overlap_vs_snr(clusters_all, clusters_separate, params)
    plot_isolation_vs_snr(clusters_all, clusters_separate, params)
    plot_isolation_vs_noise_overlap(clusters_all, clusters_separate, params)


def plot_false_positives(clusters_all, params, name, false_positives_path):
    save_output_path = params.get_save_output_path() + '/cluster_quality/'
    if os.path.exists(save_output_path) is False:
        os.makedirs(save_output_path)
    save_output_path = params.get_save_output_path() + '/cluster_quality/'

    noise_overlap = params.get_noise_overlap()
    snr = params.get_snr()

    false_pos_plot = plt.figure()
    ax = false_pos_plot.add_subplot(1, 1, 1)
    false_pos_plot, ax = plot_utility.style_plot(ax)
    x_all = clusters_all.noiseoverlap.values
    y_all = clusters_all.peakSNR.values

    false_positives_df = Published_analysis.NatComms_LocalGridHD_2020.false_positives.get_false_positives(clusters_all, false_positives_path)
    x_false_pos = false_positives_df.noiseoverlap.values
    y_false_pos = false_positives_df.peakSNR.values

    ax.plot(x_all, y_all, 'o', markersize=3, color='black', label='All accepted clusters')
    ax.plot(x_false_pos, y_false_pos, 'o', markersize=3, color='red', label='False positives')

    ax.legend(numpoints=1, frameon=False)

    plt.xlabel('Noise overlap')
    plt.ylabel('Peak signal to noise ratio')
    plt.savefig(save_output_path + 'noise_overlap_vs_snr_false_positives' + name + '.png')


def get_number_of_passing_cells(clusters, params, name, false_positives_path):
    output_path = params.get_save_output_path() + 'num_of_cells' + name + '.txt'
    out_file = open(output_path, "w")

    good_clusters = (clusters['goodcluster'] == 1)
    num_of_good_clusters = good_clusters.sum()
    print('Number of clusters that pass curation: {}'.format(num_of_good_clusters))
    out_file.write("Number of clusters that pass curation: %s" % num_of_good_clusters)


    false_positives = Published_analysis.NatComms_LocalGridHD_2020.false_positives.get_false_positives(clusters, false_positives_path)
    num_of_false_positives = false_positives.false_positive.sum()

    print('Number of false positives: {}'.format(num_of_false_positives))
    out_file.write("\nNumber of false positives: %s" % num_of_false_positives)

    passed_isolation = (clusters['isolationpass'] == 1).sum()
    print('Number of clusters that pass isolation: {}'.format(passed_isolation))
    out_file.write("\nNumber of clusters that pass isolation: %s" % passed_isolation)
    passed_snr = (clusters['peakSNRpass'] == 1).sum()
    print('Number of clusters that pass peakSNR: {}'.format(passed_snr))
    out_file.write("\nNumber of clusters that pass peakSNR: %s" % passed_snr)
    passed_noise_overlap = (clusters['noiseoverlappass'] == 1).sum()
    print('Number of clusters that pass noise overlap: {}'.format(passed_noise_overlap))
    out_file.write("\nNumber of clusters that pass noise overlap: %s" % passed_noise_overlap)

    out_file.close()


def compare_whitening(params):
    path_to_data = params.get_path_to_data()

    spike_data_frame_all = pd.read_csv(path_to_data + 'data_all.csv')  # reads csv, puts it in df
    good_cluster = spike_data_frame_all['goodcluster'] == 1
    good_clusters_df_all = spike_data_frame_all[good_cluster]

    spike_data_frame_separate = pd.read_csv(path_to_data + 'data_separate.csv')  # reads csv, puts it in df
    good_cluster = spike_data_frame_separate['goodcluster'] == 1
    good_clusters_df_separate = spike_data_frame_separate[good_cluster]

    plot_cluster_quality(spike_data_frame_all, spike_data_frame_separate, params)
    plot_false_positives(good_clusters_df_all, params, '_whitened_together', params.get_false_positives_path_all())
    plot_false_positives(good_clusters_df_separate, params, 'whitened_separately', params.get_false_positives_path_separate())

    print('All whitened together')
    get_number_of_passing_cells(spike_data_frame_all, params, '_whitened_together', params.get_false_positives_path_all())
    print('Whitened tetrode by tetrode')
    get_number_of_passing_cells(spike_data_frame_separate, params, '_whitened_separately', params.get_false_positives_path_separate())

