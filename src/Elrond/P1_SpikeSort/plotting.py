import matplotlib.pyplot as plt
import numpy as np
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
import spikeinterface.full as si

def plot_locations_over_time(recording, export_path):

    rec_length_mins = recording.get_duration()/60

    si.set_global_job_kwargs(n_jobs=4)

    peaks = detect_peaks(recording)
    peak_locations = localize_peaks(recording, peaks, method='center_of_mass')

    fs = recording.get_sampling_frequency()
    _, ax = plt.subplots(figsize=(rec_length_mins/2, 8))
    ax.scatter(peaks['sample_index'] / fs, peak_locations['y'], color='k', marker='.',  alpha=0.002)
    plt.savefig(export_path + 'locations.png')

def plot_simple_np_probe_layout(recording, export_path):
    """
    Saves a simple probe layout in `export_path` for a Neuropixels 2.0 probe.
    """

    probe = recording.get_probe()
    probe_positions = probe.contact_positions
    
    prescence_matrix = np.zeros((4,4))
    j_list = []
    i_list = []
    if 750. in probe_positions[:,0]:
        j_list.append(3)
    if 500. in probe_positions[:,0]:
        j_list.append(2)
    if 250. in probe_positions[:,0]:
        j_list.append(1)
    if 0. in probe_positions[:,0]:
        j_list.append(0)
    
    if 2160. in probe_positions[:,1]:
        i_list.append(0)
    if 1440. in probe_positions[:,1]:
        i_list.append(1)
    if 720. in probe_positions[:,1]:
        i_list.append(2)
    if 0. in probe_positions[:,1]:
        i_list.append(3)
    
    for i in i_list:
        for j in j_list:
            prescence_matrix[i,j]=1
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.set_axis_off()
    ax.imshow(prescence_matrix)

    fig.savefig(export_path + 'probe_layout.png')
