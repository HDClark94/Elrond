import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si

import Elrond.settings as settings


def trace(spike_data, output_path):
    save_path = output_path + 'spike_trace/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # Create a raster plot
    fig, ax = plt.subplots(figsize=(5,3))
    for i, cluster_id in enumerate(spike_data["cluster_id"]):
        spike_times = spike_data["firing_times"].iloc[i]/settings.sampling_rate
        shank_id = spike_data["shank_id"].iloc[i]
        if shank_id>3:
            shank_color="#6ACA47"
        else:
            shank_color="#7F81E3"
        ax.scatter(spike_times, np.ones(len(spike_times))*cluster_id, marker="|", color=shank_color, alpha=0.3)

    #ax.set_xlabel('Time (s)')
    #ax.set_ylabel('Neuron ID')
    ax.set_xlim(1000,1015)
    ax.axis("off")
    #ax.set_title('Raster Plot of Neuron Spike Times')
    plt.savefig(save_path+"spike_traces.png", dpi=500)
    plt.close()

#Test



def main():
    unit_ids = np.array([10,11,15,16,22,23,24,26, 29,
                        34,38,39,40,41,42,43,44,45,49,
                        51,52,57,59,78,86,90,91,92,93,
                        95,99,105,112,113,117,127,128,129,
                        131,133,135,139,141,142,144,145,147,
                        149,150,153,154,155,156,157,158,161,
                        162,165,166,167,168,170,171,172,174,
                        175,177,180,184,185,187,188,190,191,
                        192,194,197,198,200,201,202,205,206,
                        207,208,210,212,213, 214, 216,217,219,
                        221,222, 223, 224,225,226,228,229,230,
                        232,233, 234, 236, 237, 239, 244, 248,
                        249, 252, 260, 262, 272, 275, 276, 278,
                        279, 280, 281, 286, 287, 290])

    units_to_vis =np.array([45, 49, 51, 52, 129, 142, 149, 150, 153,157,
                            168,172,177, 198, 201,206,219, 224, 229, 239, 286])
    units_to_vis = np.array([45,142, 229, 239, 286])
    spikes = pd.read_pickle("/mnt/datastore/Harry/Cohort10_october2023/derivatives/M18/D1/vr/M18_D1_2023-10-30_12-38-29/processed/kilosort4/spikes.pkl")
    spikes = spikes[np.isin(spikes["cluster_id"], unit_ids)]
    trace(spikes, output_path="/mnt/datastore/Harry/Cohort10_october2023/derivatives/M18/D1/vr/M18_D1_2023-10-30_12-38-29/processed/kilosort4/")
    spikes = pd.read_pickle("/mnt/datastore/Harry/Cohort10_october2023/derivatives/M18/D1/of/M18_D1_2023-10-30_13-25-44/processed/kilosort4/spikes.pkl")
    spikes = spikes[np.isin(spikes["cluster_id"], unit_ids)]
    trace(spikes, output_path="/mnt/datastore/Harry/Cohort10_october2023/derivatives/M18/D1/of/M18_D1_2023-10-30_13-25-44/processed/kilosort4/")
    analyzer= si.load_sorting_analyzer("/mnt/datastore/Harry/Cohort10_october2023/derivatives/M18/D1/ephys/sorting_analyzer")

    figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)
    plotter = si.plot_unit_waveforms(analyzer, unit_ids=units_to_vis,
                                     sparsity=None, same_axis=True, ax=ax)
    ax.axis("off")
    plt.savefig("/mnt/datastore/Harry/plot_viewer/waveforms.png")

    unit_ids = units_to_vis
    shank_ids = analyzer.get_sorting_property(key="group")
    shank_colors = np.repeat("#7F81E3", len(shank_ids))
    shank_colors[shank_ids > 3] = "#6ACA47"
    shank_colors = shank_colors[np.isin(analyzer.unit_ids, unit_ids)]
    unit_colors = dict(zip(unit_ids, shank_colors))

    np.random.seed(4756)
    unit_colors_waveforms = ["purple", "red", "orange", "blue","magenta"]
    for i, id in enumerate(unit_ids):
        figsize = (2, 5)
        fig, ax = plt.subplots(figsize=figsize)
        plotter = si.plot_unit_waveforms(analyzer, unit_ids=[id], unit_colors=dict(zip([id],[unit_colors_waveforms[i]])),alpha_waveforms=0.1,
                                         sparsity=None, same_axis=True, ax=ax, max_spikes_per_unit=20, plot_legend=False, set_title=False)
        ax.axis("off")
        plt.savefig("/mnt/datastore/Harry/plot_viewer/waveforms_"+str(id)+".png", dpi=500)

    np.random.seed(4756)
    unit_colors_waveforms = ["purple", "red", "orange", "blue","magenta"]
    for i, id in enumerate(unit_ids):
        figsize = (2, 2)
        fig, ax = plt.subplots(figsize=figsize)
        plotter = si.plot_autocorrelograms(analyzer, unit_ids=[id], unit_colors=dict(zip([id],[unit_colors_waveforms[i]])), ax=ax, figtitle=None)
        ax.axis("off")
        plt.savefig("/mnt/datastore/Harry/plot_viewer/autocorrs_"+str(id)+".png", dpi=500)



    figsize=(10,10)
    fig, ax = plt.subplots(figsize=figsize)
    plotter = si.plot_crosscorrelograms(analyzer, backend="matplotlib",
                                        figure=fig,figsize=figsize, unit_ids=unit_ids,
                                        unit_colors=unit_colors)
    ax.axis("off")
    plt.savefig("/mnt/datastore/Harry/plot_viewer/plot_crosscorrelograms.png", dpi=500)


    print("================")


if __name__ == '__main__':
    main()