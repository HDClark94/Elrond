import settings
import pandas as pd
from Helpers.upload_download import *
from P1_SpikeSort.preprocess import preprocess, ammend_preprocessing_parameters
from P1_SpikeSort.probe import add_probe
from P1_SpikeSort.waveforms import extract_waveforms, get_waveforms
from P1_SpikeSort.auto_curate import auto_curate


def save_spikes_to_dataframe(sorters, waveforms, quality_metrics, recording_paths, processed_folder_name):
    for sorter, waveform, recording_path in zip(sorters, waveforms, recording_paths):
        recording_name = os.path.basename(recording_path)

        spike_data = pd.DataFrame()
        for i in sorter.get_unit_ids():
            cluster_df = pd.DataFrame()
            cluster_df['session_id'] = [recording_name]                      # str
            cluster_df['cluster_id'] = [i]                                   # int
            cluster_df['shank_id'] = [sorter.get_unit_property(i, 'group')]  # int
            cluster_df['firing_times'] = [sorter.get_unit_spike_train(i)]    # np.array(n_spikes)
            cluster_df['waveforms'] = [waveform[i]]                          # np.array(n_spikes, n_samples, n_channels)


            spike_data = pd.concat([spike_data, cluster_df], ignore_index=True)

        # add quality metrics, these are shared across all recordings
        spike_data = spike_data.merge(quality_metrics, on='cluster_id')

        processed_path = recording_path + "/" + processed_folder_name
        if not os.path.exists(processed_path):
            os.mkdir(processed_path)
        sorter_path = processed_path + "/" + settings.sorterName
        if not os.path.exists(sorter_path):
            os.mkdir(sorter_path)

        print("I am saving the spike dataframe for ", recording_path, " at ", sorter_path,  "/spikes.pkl")
        spike_data.to_pickle(sorter_path + "/spikes.pkl")


def spikesort(recording_path, local_path, processed_folder_name, **kwargs):
    # get all working paths for recordings to sort
    recording_paths = get_recordings_to_sort(recording_path, local_path, **kwargs)
    recording_formats = get_recording_formats(recording_paths)

    # load recordings to sort
    recordings = load_recordings(recording_paths, recording_formats)

    # concatenate recordings if necessary
    recording_mono = si.concatenate_recordings(recordings)

    #recording_mono = recording_mono.frame_slice(start_frame=0, end_frame=int(60*30000)) # debugging purposes

    # set probe information
    recording_mono, probe = add_probe(recording_mono, recording_path)

    # preprocess and ammend preprocessing parameters for presorting
    default_params = si.get_default_sorter_params(settings.sorterName)
    recording_mono = preprocess(recording_mono)
    params = ammend_preprocessing_parameters(default_params)

    # Run spike sorting
    sorting_mono = si.run_sorter_by_property(sorter_name=settings.sorterName,
                                             recording=recording_mono,
                                             grouping_property='group',
                                             working_folder='sorting_tmp',
                                             remove_existing_folder=True,
                                             verbose=True, **params)
    # There seems to be an warning for filtering but no filtering has been applied!
    print("Spike sorting is finished!")
    print("I found " + str(len(sorting_mono.unit_ids)) + " clusters")

    # Extract the waveforms and quality metrics from across all recordings to the extractor
    we, quality_metrics = extract_waveforms(recording_mono, sorting_mono, recording_path, processed_folder_name)
    quality_metrics["cluster_id"] = sorting_mono.get_unit_ids()

    # assign an automatic curation label based on the quality metrics
    quality_metrics = auto_curate(quality_metrics)

    # split our extractors back
    sorters = si.split_sorting(sorting_mono, recordings)
    sorters = [si.select_segment_sorting(sorters, i) for i in range(len(recordings))] # turn it into a list of sorters
    waveforms = get_waveforms(we, sorters)
    recordings = si.split_recording(recording_mono)

    # save spike times and waveform information for further analysis
    save_spikes_to_dataframe(sorters, waveforms, quality_metrics, recording_paths, processed_folder_name)

    # Optionally
    if "save2phy" in kwargs:
        if kwargs["save2phy"] == True:
            si.export_to_phy(we, output_folder=recording_path + "/" + processed_folder_name + "/" + settings.sorterName + "/phy", remove_if_exists=True, copy_binary=False)
            si.export_report(we, output_folder=recording_path + "/" + processed_folder_name + "/" + settings.sorterName + "/report", remove_if_exists=True)
