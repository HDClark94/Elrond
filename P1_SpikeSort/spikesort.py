import settings
import pandas as pd
from Helpers.upload_download import *
from P1_SpikeSort.preprocess import preprocess, ammend_preprocessing_parameters
from P1_SpikeSort.probe import add_probe
from P1_SpikeSort.waveforms import extract_waveforms, get_waveforms
from P1_SpikeSort.auto_curate import auto_curate

si.set_global_job_kwargs(n_jobs=-1)

def save_spikes_to_dataframe(sorters, recordings, quality_metrics,
                             recording_paths, processed_folder_name, sorterName, spike_data=None):

    for sorter, recording, recording_path in zip(sorters, recordings, recording_paths):
        recording_name = os.path.basename(recording_path)

        new_spike_data = pd.DataFrame()
        for i, id in enumerate(sorter.get_unit_ids()):
            cluster_df = pd.DataFrame()
            cluster_df['session_id'] = [recording_name]                      # str
            cluster_df['cluster_id'] = [id]                                  # int
            cluster_df['firing_times'] = [sorter.get_unit_spike_train(id)]   # np.array(n_spikes)
            cluster_df['mean_firing_rate'] = [len(sorter.get_unit_spike_train(id))/recording.get_duration()]
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
    recording_mono = preprocess(recording_mono)
    recording_mono, probe = add_probe(recording_mono, recording_path)
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
    sorters = [si.select_segment_sorting(sorters, i) for i in range(len(recordings))] # turn it into a list of sorters

    # save spike times and waveform information for further analysis
    save_spikes_to_dataframe(sorters, recordings, quality_metrics,
                             recording_paths, processed_folder_name, sorterName, spike_data=spike_data)

    # Optionally
    si.export_report(we, output_folder=recording_path + "/" + processed_folder_name + "/" + sorterName +
                                               "/manually_curated_report", remove_if_exists=True)
    return


def spikesort(
        recording_path, 
        local_path, 
        processed_folder_name, 
        do_spike_sorting = True,
        do_spike_postprocessing = True,
        make_report = True,
        make_phy_output = True,
        curate_using_phy = True,
        auto_curate = False,
        sorting_analyzer_path = None,
        phy_path = None,
        report_path = None,
        **kwargs
    ):

    if "sorterName" in kwargs:
        sorterName = kwargs["sorterName"]
    else:
        sorterName = settings.sorterName
    

    if sorting_analyzer_path is None:
        sorting_analyzer_path = recording_path + "/" + processed_folder_name + "/" + sorterName + "/sorting_analyzer"
    if phy_path is None:
        phy_path = recording_path+"/"+processed_folder_name +"/"+sorterName+"/phy"
    if report_path is None:
        report_path = recording_path + "/" + processed_folder_name + "/" + sorterName + "/uncurated_report"

    # create recording extractor
    recording_paths = get_recordings_to_sort(recording_path, local_path, **kwargs)
    recording_formats = get_recording_formats(recording_paths)
    recordings = load_recordings(recording_paths, recording_formats)
    recording_mono = si.concatenate_recordings(recordings)
    recording_mono, probe = add_probe(recording_mono, recording_path)

    #recording_mono = recording_mono.frame_slice(start_frame=0, end_frame=int(30 * 30000))  # debugging purposes

    # preprocess and ammend preprocessing parameters for presorting
    params = si.get_default_sorter_params(sorterName)
    recording_mono = preprocess(recording_mono)
    params = ammend_preprocessing_parameters(params, **kwargs)

    if do_spike_sorting:
        print("I will sort using", sorterName)
        # Run spike sorting
        sorting_mono = si.run_sorter_by_property(
            sorter_name=sorterName,
            recording=recording_mono,
            grouping_property='group',
            folder='sorting_tmp',
            remove_existing_folder=True,
            verbose=False, 
            **params
        )
        # There seems to be an warning for filtering but no filtering has been applied!
        print("Spike sorting is finished!")
        print("I found " + str(len(sorting_mono.unit_ids)) + " clusters")

        # make sorting analyzer
        
        sorting_analyzer = si.create_sorting_analyzer(
            sorting = sorting_mono, 
            recording=recording_mono,
            format="binary_folder",
            folder=sorting_analyzer_path
        )
    else:
        sorting_analyzer = si.load_sorting_analyzer(sorting_analyzer_path)
        sorting_analyzer._recording = recording_mono

    if do_spike_postprocessing or make_phy_output or make_report:

        # compute sorting analyzer extensions. Kwargs go in {}
        sorting_analyzer.compute({
            "random_spikes": {},
            "noise_levels": {},
            "waveforms": {},
            "templates": {},
            "correlograms": {},
            "spike_locations": {},
            "spike_amplitudes": {},
            "quality_metrics": {},
            "template_similarity": {},
            "template_metrics": {}
        })

    if auto_curate:

        quality_metrics = sorting_analyzer.get_extension("quality_metrics").get_data()
        quality_metrics['cluster_id'] = sorting_analyzer.sorting_mono.get_unit_ids()

        # assign an automatic curation label based on the quality metrics
        quality_metrics = auto_curate(quality_metrics)

    # this should replace the update_from_phy function - test when we do manual curation
    if curate_using_phy:
        sorting_mono = si.read_phy(phy_path, exclude_cluster_groups=["noise", "mua"])
        sorting_analyzer._sorting = sorting_mono

    if make_phy_output and not curate_using_phy:
        si.export_to_phy(sorting_analyzer, output_folder=phy_path, remove_if_exists=True, copy_binary=True)
    if make_report:
        si.export_report(sorting_analyzer, output_folder=report_path, remove_if_exists=True)
        

    # split our extractors back
    sorters = si.split_sorting(sorting_mono, recordings)
    sorters = [si.select_segment_sorting(sorters, i) for i in range(len(recordings))] # turn it into a list of sorters

    # save spike times and waveform information for further analysis
    save_spikes_to_dataframe(sorters, recordings, quality_metrics, recording_paths, processed_folder_name, sorterName)

    return