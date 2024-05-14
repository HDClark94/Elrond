import settings
import pandas as pd
from Helpers.upload_download import *
from P1_SpikeSort.preprocess import preprocess, ammend_preprocessing_parameters
from P1_SpikeSort.probe import add_probe
from P1_SpikeSort.waveforms import extract_waveforms, get_waveforms
from P1_SpikeSort.auto_curate import auto_curate


def save_spikes_to_dataframe(sorters, waveforms, quality_metrics, recording_paths, processed_folder_name, sorterName, spike_data=None):
    for sorter, waveform, recording_path in zip(sorters, waveforms, recording_paths):
        recording_name = os.path.basename(recording_path)

        new_spike_data = pd.DataFrame()
        for i, id in enumerate(sorter.get_unit_ids()):
            cluster_df = pd.DataFrame()
            cluster_df['session_id'] = [recording_name]                      # str
            cluster_df['cluster_id'] = [id]                                   # int
            cluster_df['firing_times'] = [sorter.get_unit_spike_train(id)]    # np.array(n_spikes)
            cluster_df['waveforms'] = [waveform[i]]                          # np.array(n_spikes, n_samples, n_channels)
            if spike_data is not None:
                cluster_df['shank_id'] = [spike_data[spike_data["cluster_id"] == id]['shank_id'].iloc[0]] # int
            else:
                cluster_df['shank_id'] = [sorter.get_unit_property(id, 'group')]  # int

            new_spike_data = pd.concat([new_spike_data, cluster_df], ignore_index=True)

        # add quality metrics, these are shared across all recordings
        new_spike_data = new_spike_data.merge(quality_metrics, on='cluster_id')

        processed_path = recording_path + "/" + processed_folder_name
        if not os.path.exists(processed_path):
            os.mkdir(processed_path)
        sorter_path = processed_path + "/" + sorterName
        if not os.path.exists(sorter_path):
            os.mkdir(sorter_path)

        print("I am saving the spike dataframe for ", recording_path, " at ", sorter_path,  "/spikes.pkl")
        new_spike_data.to_pickle(sorter_path + "/spikes.pkl")


def update_from_phy(recording_path, local_path, processed_folder_name, **kwargs):
    # create recording extractor
    recording_paths = get_recordings_to_sort(recording_path, local_path, **kwargs)
    recording_formats = get_recording_formats(recording_paths)
    recordings = load_recordings(recording_paths, recording_formats)
    recording_mono = si.concatenate_recordings(recordings)
    recording_mono, probe = add_probe(recording_mono, recording_path)
    recording_mono = preprocess(recording_mono)

    if "sorterName" in kwargs:
        sorterName = kwargs["sorterName"]
    else:
        sorterName = settings.sorterName
    print("I will use sorted results from", sorterName)

    spike_data = pd.read_pickle(recording_paths[0]+"/"+processed_folder_name +"/"+sorterName+"/spikes.pkl")

    # create sorting extractor from first primarly recording phy folder
    phy_path = recording_paths[0]+"/"+processed_folder_name +"/"+sorterName+"/phy"
    sorting_mono = si.read_phy(phy_path, exclude_cluster_groups=["noise", "mua"])
    print("I found " + str(len(sorting_mono.unit_ids)) + " clusters")

    # Extract the waveforms and quality metrics from across all recordings to the extractor
    we, quality_metrics = extract_waveforms(recording_mono, sorting_mono, recording_path, processed_folder_name, sorterName)
    quality_metrics["cluster_id"] = sorting_mono.get_unit_ids()

    # assign a new automatic curation label based on the quality metrics
    quality_metrics = auto_curate(quality_metrics)

    # split our extractors back
    sorters = si.split_sorting(sorting_mono, recordings)
    sorters = [si.select_segment_sorting(sorters, i) for i in range(len(recordings))]  # turn it into a list of sorters
    waveforms = get_waveforms(we, sorters)
    recordings = si.split_recording(recording_mono)

    # save spike times and waveform information for further analysis
    save_spikes_to_dataframe(sorters, waveforms, quality_metrics, recording_paths, processed_folder_name, spike_data=spike_data)

    # Optionally
    if "save2phy" in kwargs:
        if kwargs["save2phy"] == True:
            si.export_report(we,
                             output_folder=recording_path + "/" + processed_folder_name + "/" + sorterName + "/manually_curated_report",
                             remove_if_exists=True)
    return


def spikesort(recording_path, local_path, processed_folder_name, **kwargs):
    # create recording extractor
    recording_paths = get_recordings_to_sort(recording_path, local_path, **kwargs)
    recording_formats = get_recording_formats(recording_paths)
    recordings = load_recordings(recording_paths, recording_formats)
    recording_mono = si.concatenate_recordings(recordings)
    recording_mono, probe = add_probe(recording_mono, recording_path)

    #recording_mono = recording_mono.frame_slice(start_frame=0, end_frame=int(30 * 30000))  # debugging purposes

    if "sorterName" in kwargs:
        sorterName = kwargs["sorterName"]
    else:
        sorterName = settings.sorterName
    print("I will sort using", sorterName)

    # preprocess and ammend preprocessing parameters for presorting
    params = si.get_default_sorter_params(sorterName)
    recording_mono = preprocess(recording_mono)
    params = ammend_preprocessing_parameters(params)

    # note for using kilosort4 https://github.com/MouseLand/Kilosort/issues/606
    # suggests ammending these parameters but error still occurs with all NP recordings as of 12/05/2024
    # params["dminx"] = 400; params["nearest_templates"] = 20

    # Run spike sorting
    sorting_mono = si.run_sorter_by_property(sorter_name=sorterName,
                                             recording=recording_mono,
                                             grouping_property='group',
                                             working_folder='sorting_tmp',
                                             remove_existing_folder=True,
                                             verbose=True, **params)
    # There seems to be an warning for filtering but no filtering has been applied!
    print("Spike sorting is finished!")
    print("I found " + str(len(sorting_mono.unit_ids)) + " clusters")

    # Extract the waveforms and quality metrics from across all recordings to the extractor
    we, quality_metrics = extract_waveforms(recording_mono, sorting_mono, recording_path, processed_folder_name, sorterName)
    quality_metrics["cluster_id"] = sorting_mono.get_unit_ids()

    # assign an automatic curation label based on the quality metrics
    quality_metrics = auto_curate(quality_metrics)

    # split our extractors back
    recordings = si.split_recording(recording_mono)
    sorters = [si.select_segment_sorting(sorting_mono, i) for i in range(len(recordings))] # turn it into a list of sorters
    waveforms = get_waveforms(we, sorters)

    # save spike times and waveform information for further analysis
    save_spikes_to_dataframe(sorters, waveforms, quality_metrics, recording_paths, processed_folder_name, sorterName)

    # Optionally
    if "save2phy" in kwargs:
        if kwargs["save2phy"] == True:
            si.export_to_phy(we, output_folder=recording_path + "/" + processed_folder_name + "/" + sorterName + "/phy", remove_if_exists=True, copy_binary=True)
            si.export_report(we, output_folder=recording_path + "/" + processed_folder_name + "/" + sorterName + "/uncurated_report", remove_if_exists=True)
    return