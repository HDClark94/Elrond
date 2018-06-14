import os

import matplotlib.pylab as plt
import numpy as np

from PreClustering import open_ephys_IO


# This is necessary because of artefacts at the beginning and end of the recordings. 5 seconds are removed
def remove_beginning_and_end(prm, raw_light_data):
    sampling_rate = prm.get_sampling_rate()
    raw_light_trimmed = raw_light_data[sampling_rate*5:-sampling_rate*5]
    return raw_light_trimmed


def save_or_open_light(prm):
    file_path = prm.get_filepath() + prm.get_opto_ch()
    light_intensity = open_ephys_IO.get_data_continuous(prm, file_path)
    light_intensity = remove_beginning_and_end(prm, light_intensity)

    light_on = np.where(light_intensity > 2.5)
    np.save(prm.get_filepath() + 'light_on_indices', light_on)
    # plt.plot(light_intensity)
    # plt.plot(light_on, 3.27, 'ro')
    # plt.show()

    return light_intensity


def process_opto(prm):
    folder_path = prm.get_filepath()
    file_path = prm.get_filepath() + prm.get_opto_ch()
    if os.path.isfile(prm.get_filepath() + 'light_on_indices') is False:
        save_or_open_light(prm)


def plot_light_responses(prm, spike_times_of_unit):
    light_on_indices = np.load(prm.get_filepath() + 'light_on_indices.npy')
    plt.plot(light_on_indices)
    plt.show()
