import Elrond.Helpers.OpenEphys as OpenEphys

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
