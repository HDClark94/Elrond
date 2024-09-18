##some helper functions to be used with spike interface

import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as sorters
import spikeinterface.preprocessing as spre
from spikeinterface.postprocessing import compute_spike_amplitudes, compute_principal_components
from spikeinterface.exporters import export_to_phy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .file_utility import *
import settings as settings
from probeinterface.plotting import plot_probe
from probeinterface import get_probe
from probeinterface import Probe, ProbeGroup
from .OpenEphys import *

def load_OpenEphysRecording(folder, channel_ids=None):
    number_of_channels, corrected_data_file_suffix = count_files_that_match_in_folder(folder, data_file_prefix=settings.data_file_prefix, data_file_suffix='.continuous')
    if channel_ids is None:
        channel_ids = np.arange(1, number_of_channels+1)

    signal = []
    for i, channel_id in enumerate(channel_ids):
        fname = folder+'/'+corrected_data_file_suffix+str(channel_id)+settings.data_file_suffix+'.continuous'
        x = OpenEphys.loadContinuousFast(fname)['data']
        if i==0:
            #preallocate array on first run
            signal = np.zeros((x.shape[0], len(channel_ids)))
        signal[:,i] = x
    return [signal]


def test_probe_interface(save_path):
    num_channels = 64
    recording, sorting_true = se.toy_example(duration=5, num_channels=num_channels, seed=0, num_segments=4)

    other_probe = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
    print(other_probe)

    other_probe.set_device_channel_indices(np.arange(num_channels))
    recording_4_shanks = recording.set_probe(other_probe, group_mode='by_shank')
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1)
    plot_probe(recording_4_shanks.get_probe(), ax=ax)
    plt.savefig(save_path+'probe_locations.png', dpi=200)
    plt.close()

def add_probe_info_by_shank(recordingExtractor, shank_df):
    x = shank_df["x"].values
    y = shank_df["y"].values
    geom = np.transpose(np.vstack((x,y)))
    probe = Probe(ndim=2, si_units='um')
    if shank_df["contact_shapes"].iloc[0] == "rect":
        probe.set_contacts(positions=geom, shapes='rect', shape_params={'width': shank_df["width"].iloc[0],
                                                                        'height': shank_df["height"].iloc[0]})
    elif shank_df["contact_shapes"].iloc[0] == "circle":
        probe.set_contacts(positions=geom, shapes='circle', shape_params={'radius': shank_df["radius"].iloc[0]})
    #probe.set_device_channel_indices(np.array(shank_df["channel"]))
    probe.set_device_channel_indices(np.arange(len(shank_df)))
    recordingExtractor.set_probe(probe, in_place=True)
    return recordingExtractor


def run_spike_sorting_with_spike_interface(recording_to_sort, sorterName):
    # load signal
    '''
    base_signal = load_OpenEphysRecording(recording_to_sort)
    base_recording = se.NumpyRecording(base_signal,settings.sampling_rate)
    base_recording = add_probe_info(base_recording, recording_to_sort, sorterName)
    base_recording = spre.whiten(base_recording)
    base_recording = spre.bandpass_filter(base_recording, freq_min=300, freq_max=6000)
    bad_channel_ids = getDeadChannel(recording_to_sort +'/dead_channels.txt')
    base_recording.remove_channels(bad_channel_ids)
    base_recording = base_recording.save(folder= settings.temp_storage_path+'/processed', n_jobs=1, chunk_size=2000, progress_bar=True, overwrite=True)

    params = sorters.get_default_sorter_params(sorterName)
    params['filter'] = False #have already done this in preprocessing step
    params['whiten'] = False
    params['adjacency_radius'] = 200

    sorting = sorters.run_sorter(sorter_name=sorterName, recording=base_recording,output_folder='sorting_tmp', verbose=True, **params)
    sorting = sorting.save(folder= settings.temp_storage_path+'/sorter', n_jobs=1, chunk_size=2000, progress_bar=True, overwrite=True)
    '''

    params = sorters.get_default_sorter_params(sorterName)
    params['filter'] = False #have already done this in preprocessing step
    params['whiten'] = False
    params['adjacency_radius'] = 200
    params['num_workers'] = 3

    n_channels, _ = count_files_that_match_in_folder(recording_to_sort, data_file_prefix=settings.data_file_prefix, data_file_suffix='.continuous')
    probe_group_df = get_probe_dataframe(n_channels)
    #probe_group = generate_probe_group(n_channels)
    #bad_channel_ids = getDeadChannel(recording_to_sort +'/dead_channels.txt')

    for probe_index in np.unique(probe_group_df["probe_index"]):
        print("I am subsetting the recording and analysing probe "+str(probe_index))
        probe_df = probe_group_df[probe_group_df["probe_index"] == probe_index]
        for shank_index in np.unique(probe_df["shank_ids"]):
            print("I am looking at shank "+str(shank_index))
            shank_df = probe_df[probe_df["shank_ids"] == shank_index]
            channels_in_shank = np.array(shank_df["channel"])
            signal_shank = load_OpenEphysRecording(recording_to_sort, channel_ids=channels_in_shank)
            shank_recording = se.NumpyRecording(signal_shank, settings.sampling_rate)
            shank_recording = add_probe_info_by_shank(shank_recording, shank_df)
            shank_recording = spre.whiten(shank_recording)
            shank_recording = spre.bandpass_filter(shank_recording, freq_min=300, freq_max=6000)

            shank_recording = shank_recording.save(folder= settings.temp_storage_path+'/processed_probe'+str(probe_index)+'_shank'+str(shank_index)+'_segment0',
                                                             n_jobs=1, chunk_size=2000, progress_bar=True, overwrite=True)
            shank_sorting = sorters.run_sorter(sorter_name=sorterName, recording=shank_recording, output_folder='sorting_tmp', verbose=True, **params)
            shank_sorting = shank_sorting.save(folder= settings.temp_storage_path+'/sorter_probe'+str(probe_index)+'_shank'+str(shank_index)+'_segment0',
                                               n_jobs=1, chunk_size=2000, progress_bar=True, overwrite=True)

            we = si.extract_waveforms(shank_recording, shank_sorting, folder=settings.temp_storage_path+'/waveforms_probe'+str(probe_index)+'_shank'+str(shank_index)+'_segment0',
                                      ms_before=1, ms_after=2, load_if_exists=False, overwrite=True, return_scaled=False)

            on_shank_cluster_ids = shank_sorting.get_unit_ids()
            cluster_ids = get_probe_shank_cluster_ids(on_shank_cluster_ids, probe_id=probe_index, shank_id=shank_index)

            _ = compute_spike_amplitudes(waveform_extractor=we)
            _ = compute_principal_components(waveform_extractor=we, n_components=3, mode='by_channel_global')
            save_to_phy(we, settings.temp_storage_path+'/phy_folder', probe_index=probe_index, shank_index=shank_index)
            save_waveforms_locally(we, settings.temp_storage_path+'/waveform_arrays/', on_shank_cluster_ids, cluster_ids, segment=0)
    return

def get_on_shank_cluster_ids(cluster_ids):
    on_shank_cluster_ids = []
    for i in range(len(cluster_ids)):
        on_shank_cluster_id = str(int(cluster_ids[i]))
        on_shank_cluster_id = on_shank_cluster_id[2:]
        on_shank_cluster_ids.append(int(on_shank_cluster_id))
    return on_shank_cluster_ids

def get_probe_shank_cluster_ids(on_shank_cluster_ids, probe_id, shank_id):
    cluster_ids = []
    for i in range(len(on_shank_cluster_ids)):
        on_shank_cluster_id = int(on_shank_cluster_ids[i])
        cluster_id = str(probe_id)+str(shank_id)+str(on_shank_cluster_id)
        cluster_ids.append(cluster_id)
    return cluster_ids

def save_waveforms_locally(we, save_folder_path, on_shank_cluster_ids, cluster_ids, segment):
    if os.path.exists(save_folder_path) is False:
        os.makedirs(save_folder_path)
    for on_shank_id, cluster_id in zip(on_shank_cluster_ids, cluster_ids):
        waveforms = we.get_waveforms(unit_id=on_shank_id)
        np.save(save_folder_path+"waveforms_"+str(int(cluster_id))+"_segment"+str(segment)+".npy", np.array(waveforms))
    return

def save_to_phy(we, save_folder_path, probe_index, shank_index):
    if os.path.exists(save_folder_path) is False:
        os.makedirs(save_folder_path)
    shank_specific_path = save_folder_path+"/probe"+str(probe_index)+"_shank"+str(shank_index)
    export_to_phy(waveform_extractor=we, output_folder=shank_specific_path)

def generate_probe_group(number_of_channels):

    probegroup = ProbeGroup()
    if number_of_channels == 16: # presume tetrodes
        geom = pd.read_csv(settings.tetrode_geom_path, header=None).values
        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=geom, shapes='circle', shape_params={'radius': 5})
        probe.set_device_channel_indices(np.arange(number_of_channels))
    else:
        # presume cambridge neurotech P1 probes
        assert number_of_channels % 64 == 0
        n_probes = int(number_of_channels / 64)
        #device_channel_indices = []
        for i in range(n_probes):
            probe = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
            probe.wiring_to_device('cambridgeneurotech_mini-amp-64', channel_offset=int(i * 64))
            probe.move([i * 2000, 0])  # move the probes far away from eachother
            probe.set_contact_ids(np.array(probe.to_dataframe()["contact_ids"].values, dtype=np.int64) + int(64 * i))
            probe.set_device_channel_indices(np.arange(64) + int(64 * i))
            probegroup.add_probe(probe)
    return probegroup

def get_probe_dataframe(number_of_channels):
    if number_of_channels == 16: # presume tetrodes
        geom = pd.read_csv(settings.tetrode_geom_path, header=None).values
        probe = Probe(ndim=2, si_units='um')
        probe.set_contacts(positions=geom, shapes='circle', shape_params={'radius': 5})
        probe.set_device_channel_indices(np.arange(number_of_channels))
        probe_df = probe.to_dataframe()
        probe_df["channel"] = np.arange(1,16+1)
        probe_df["shank_ids"] = 1
        probe_df["probe_index"] = 1

    else: # presume cambridge neurotech P1 probes
        assert number_of_channels%64==0
        probegroup = ProbeGroup()
        n_probes = int(number_of_channels/64)
        device_channel_indices = []
        for i in range(n_probes):
            probe = get_probe('cambridgeneurotech', 'ASSY-236-P-1')
            probe.wiring_to_device('cambridgeneurotech_mini-amp-64', channel_offset=int(i * 64))
            probe.move([i * 2000, 0])  # move the probes far away from eachother
            probe.set_contact_ids(np.array(probe.to_dataframe()["contact_ids"].values, dtype=np.int64) + int(64 * i))
            probegroup.add_probe(probe)
            # TODO IS THIS RIGHT?
            device_channel_indices.extend(probe.device_channel_indices.tolist())

        device_channel_indices = np.array(device_channel_indices)+1
        probe_df = probegroup.to_dataframe()
        probe_df = probe_df.astype({"probe_index": int, "shank_ids": int, "contact_ids": int})
        probe_df["channel"] = device_channel_indices.tolist()
        probe_df["shank_ids"] = (np.asarray(probe_df["shank_ids"])+1).tolist()
        probe_df["probe_index"] = (np.asarray(probe_df["probe_index"])+1).tolist()
    return probe_df

# source: https://stackoverflow.com/users/48956/user48956
def df_empty(columns, dtypes, index=None):
    assert len(columns) == len(dtypes)
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df

def sorter2dataframe(sorter, session_id, probe_id, shank_id):
    """Convert data in a sorter to dataframe
        Arguments:
        sorter {SortingExtractor} -- the sorter to extract data frome
        session_id {str} -- ID of the session
    
    Returns:
        pandas.DataFrame -- dataframe containing the spike train, unit featuers and spike features of the sorter
    """

    df = df_empty(['session_id', 'probe_id', 'shank_id', 'cluster_id', 'firing_times'],
                  dtypes=[np.str0, np.str0, np.str0, np.uint64, np.uint64])

    # cluster ids identified as probe_id, shank_id, cluster_id 1214 is probe 1, shank 2, cluster 14.
    # This assumes there is less than 10 probes and less than 10 shanks per probe
    for i in sorter.get_unit_ids():
        cell_df = pd.DataFrame()
        cell_df['session_id'] = [session_id]
        cell_df['probe_id'] = [probe_id]
        cell_df['shank_id'] = [shank_id]
        cell_df['cluster_id'] = [int(str(probe_id)+str(shank_id)+str(i))]
        cell_df['firing_times'] = [sorter.get_unit_spike_train(i)]
        df = pd.concat([df, cell_df], ignore_index=True)
    return df
 