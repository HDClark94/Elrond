import numpy as np
import matplotlib.pyplot as plt
import os
import traceback
import warnings
import sys
from Helpers.upload_download import *
import pandas as pd
import neo
import quantities as pq
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cross_correlation_histogram
from elephant.spike_train_surrogates import randomise_spikes

def plot_cross_correlogram(spike_data, save_path, time_window_ms=500, perform_shuffle=False):
    time_window_half_ms = int(time_window_ms/2)
    colors = ['darkturquoise', 'salmon', u'#2ca02c', u'#d62728',
              u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    spike_trains = []
    binned_spike_trains = []
    for i, id in enumerate(spike_data["cluster_id"]):
        firing_times = spike_data.iloc[i]["firing_times"]/settings.sampling_rate
        firing_times = firing_times[firing_times<int(30*60)] # only take 30 minutes
        st = neo.SpikeTrain(firing_times, t_start=0, t_stop=int(30*60), units='s')
        bst = BinnedSpikeTrain(st, bin_size=1*pq.ms)
        spike_trains.append(st)
        binned_spike_trains.append(bst)

    for i, i_id in enumerate(spike_data["cluster_id"]):
        for j, j_id in enumerate(spike_data["cluster_id"]):
            fig, ax = plt.subplots(figsize=(6,3))
            #use this to color text
            shank_id_i = int(spike_data.iloc[i]["shank_id"])
            shank_id_j = int(spike_data.iloc[j]["shank_id"])

            if len(spike_trains[i])>0 and len(spike_trains[j]>0):
                cch, lags = cross_correlation_histogram(binned_spike_trains[i], binned_spike_trains[j],
                                                        window=[-time_window_half_ms, time_window_half_ms],
                                                        border_correction=False, binary=False, kernel=None)

                ax.bar(lags, cch.flatten(), color="black", width=1)
                ax.text(0.05, 0.95, "c" + str(i_id), ha='left', va='top', color=colors[shank_id_i], transform=ax.transAxes, fontsize=15)
                ax.text(0.9, 0.95, "c" + str(j_id), ha='left', va='top', color=colors[shank_id_j], transform=ax.transAxes, fontsize=15)
                ax.text(0.05, 0.85, "s" + str(shank_id_i), ha='left', va='top', color=colors[shank_id_i], transform=ax.transAxes, fontsize=15)
                ax.text(0.9, 0.85, "s" + str(shank_id_j), ha='left', va='top', color=colors[shank_id_j], transform=ax.transAxes, fontsize=15)
                ax.set_xlim(-time_window_half_ms, time_window_half_ms)


                if perform_shuffle:
                    shuffle_spike_trains = randomise_spikes(spike_trains[j], n_surrogates=100)
                    peaks = []
                    for shuffle in shuffle_spike_trains:
                        shuffled_binned_spike_train = BinnedSpikeTrain(shuffle, bin_size=1 * pq.ms)
                        cch_shuffle, _ = cross_correlation_histogram(binned_spike_trains[i], shuffled_binned_spike_train,
                                         window=[-time_window_half_ms, time_window_half_ms],
                                         border_correction=False, binary=False, kernel=None)
                        peaks.append(max(cch_shuffle.flatten()))
                    ax.axhline(np.nanpercentile(peaks, 95), linestyle="dashed", color="red", linewidth=0.7)
                    ax.axhline(np.nanpercentile(peaks, 99), linestyle="dotted", color="red", linewidth=0.7)

                plt.savefig(save_path + '/cross_correlogram_c'+str(i_id)+'_c'+str(j_id)+'.png', dpi=200)
                plt.close()

            else:
                print("no spikes from one of the two trains")
    return

def process_recordings(recording_paths, processed_folder_name, sorterName):
    for recording_path in recording_paths:
        try:
            recording_name = os.path.basename(recording_path)
            matched_recording_path = get_matched_recording_paths(recording_path)

            if os.path.exists(recording_path+"/"+processed_folder_name+"/"+sorterName+"/spikes.pkl"):
                spike_data = pd.read_pickle(recording_path+"/"+processed_folder_name+"/"+sorterName+"/spikes.pkl")
                plot_cross_correlogram(spike_data,save_path=recording_path + "/" + processed_folder_name + "/" + sorterName + "/plots/correlograms")

            #if os.path.exists(matched_recording_path[0]+"/"+processed_folder_name+"/"+sorterName+"/spikes.pkl"):
            #    matched_spike_data = pd.read_pickle(matched_recording_path[0]+"/"+processed_folder_name+"/"+sorterName+"/spikes.pkl")

        except Exception as ex:
            print('There was a problem! This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
    return


def main():
    print("===============================")
    recording_paths = ["/mnt/datastore/Harry/test_recording/vr/M18_D1_2023-10-30_12-38-29"]
    recording_paths =["/mnt/datastore/Harry/Cohort9_february2023/vr/M16_D1_2023-02-28_17-42-27"]
    recording_paths = ["/mnt/datastore/Harry/Cohort9_february2023/of/M16_D1_2023-02-28_18-42-28"]
    process_recordings(recording_paths, processed_folder_name="processed", sorterName="mountainsort4")


if __name__ == '__main__':
    main()