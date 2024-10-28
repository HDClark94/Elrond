import tables
import matplotlib.pylab as plt
import numpy as np
from Helpers import plot_utility


def get_snippets(filename, path_to_data):
    path = path_to_data + filename + '/' + 'Firings0.mat'
    firings = tables.openFile(path)
    cluster_id = firings.root.cluid[:]
    cluster_id = cluster_id.flatten()
    spike_index = firings.root.spikeind[:]
    spike_index = spike_index.flatten()
    waveforms = firings.root.waveforms[:]
    return cluster_id, spike_index, waveforms


def plot_interesting_cells(cluster_id, spike_index, save_output_path):
    cluster5 = np.where(cluster_id == 5)
    cluster5_firings = np.take(spike_index, cluster5[0])
    cluster6 = np.where(cluster_id == 6)
    cluster6_firings = np.take(spike_index, cluster6[0])

    c5 = np.array(cluster5_firings, dtype=int)
    c6 = np.array(cluster6_firings, dtype=int)

    spikes_fig_whole_recording = plt.figure()
    ax1 = spikes_fig_whole_recording.add_subplot(1, 1, 1)
    spikes_fig_whole_recording, ax1 = plot_utility.style_plot(ax1)
    ax1.get_yaxis().set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.vlines(c5, 0.5, 0.95, color='navy')
    ax1.vlines(c6, 1, 1.45, color='black')
    plt.title('C5 and C6 from M5, 6th of March')
    plt.xlabel('sample points')
    plt.ylabel('cluster')
    plt.savefig(save_output_path + 'M5_conjunctive_vs_grid_cluster')

    spikes_small_section = plt.figure()
    ax2 = spikes_small_section.add_subplot(1, 1, 1)
    spikes_small_section, ax2 = plot_utility.style_plot(ax2)
    ax2.get_yaxis().set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.vlines(c5, 0.5, 0.95, color='navy')
    ax2.vlines(c6, 1, 1.45, color='black')
    plt.title('C5 and C6 from M5, 6th of March')
    plt.xlabel('sample points')
    plt.ylabel('cluster')
    plt.xlim(3050000, 3150000)
    plt.savefig(save_output_path + 'M5_conjunctive_vs_grid_cluster_section')


def analyze_snippets(dataframe, path_to_data, save_output_path):
    cluster_id, spike_index, waveforms = get_snippets('M5_2018-03-06_15-34-44_of', path_to_data)
    plot_interesting_cells(cluster_id, spike_index, save_output_path)



