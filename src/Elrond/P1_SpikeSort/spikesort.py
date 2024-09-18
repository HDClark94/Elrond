import pandas as pd
from ..Helpers.upload_download import *
from .preprocess import preprocess, ammend_preprocessing_parameters
from ..P1_SpikeSort.auto_curate import auto_curation
from ..P1_SpikeSort.probe import add_probe 
import settings as settings
from datetime import datetime

from os.path import expanduser
si.set_global_job_kwargs(n_jobs=1)

def save_spikes_to_dataframe(sorters, quality_metrics,
                             rec_samples, recording_paths, processed_paths, sorterName):

    for sorter, rec_sample, processed_path, recording_path in zip(sorters, rec_samples, processed_paths, recording_paths):
        recording_name = recording_path.split('/')[-1]

        new_spike_data = pd.DataFrame()
        for i, id in enumerate(sorter.get_unit_ids()):
            cluster_df = pd.DataFrame()
            cluster_df['session_id'] = [recording_name]                      # str
            cluster_df['cluster_id'] = [id]                                  # int
            cluster_df['firing_times'] = [sorter.get_unit_spike_train(id)]   # np.array(n_spikes)
            cluster_df['mean_firing_rate'] = [len(sorter.get_unit_spike_train(id))/(rec_sample/30000)]
            cluster_df['shank_id'] = [sorter.get_unit_property(id, 'group')]  # int
            new_spike_data = pd.concat([new_spike_data, cluster_df], ignore_index=True)

        # add quality metrics, these are shared across all recordings
        new_spike_data = new_spike_data.merge(quality_metrics, on='cluster_id')

        pkl_folder = processed_path + sorterName + "/"
        print(pkl_folder)
        Path(pkl_folder).mkdir(parents=True, exist_ok=True)
        print("I am saving the spike dataframe for ", recording_path, " in ", pkl_folder)
        new_spike_data.to_pickle(pkl_folder + "spikes.pkl")


def spikesort(
        recording_paths,
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
        processed_paths = None,
        **kwargs
    ):

    if "sorterName" in kwargs:
        sorterName = kwargs["sorterName"]
    else:
        sorterName = settings.sorterName

    if sorting_analyzer_path is None:
        sorting_analyzer_path = recording_paths[0] + "/" + processed_folder_name + "/" + sorterName + "/sorting_analyzer"
    if phy_path is None:
        phy_path = recording_paths[0]+"/"+processed_folder_name +"/"+sorterName+"/phy"
    if report_path is None:
        report_path = recording_paths[0] + "/" + processed_folder_name + "/" + sorterName + "/uncurated_report"

    recording_mono, rec_samples = make_recording_from_paths_and_get_times(recording_paths)

    # preprocess and ammend preprocessing parameters for presorting
    params = si.get_default_sorter_params(sorterName)
    recording_mono = preprocess(recording_mono)
    params = ammend_preprocessing_parameters(params, **kwargs)


    print("Before spike sorting, time is", datetime.now())
    sorting_mono=None
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
        sorting_analyzer = si.load_sorting_analyzer(sorting_analyzer_path, load_extensions=True)
        sorting_analyzer._recording = recording_mono

    print("Before postprocessing, time is", datetime.now())

    if do_spike_postprocessing: 

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

    quality_metrics = sorting_analyzer.get_extension("quality_metrics").get_data()
    quality_metrics['cluster_id'] = sorting_analyzer.sorting.get_unit_ids()

    if auto_curate:

        # assign an automatic curation label based on the quality metrics
        quality_metrics = auto_curation(quality_metrics)

    # this should replace the update_from_phy function - test when we do manual curation
    if curate_using_phy:
        sorting_mono = si.read_phy(phy_path, exclude_cluster_groups=["noise", "mua"])
        sorting_analyzer._sorting = sorting_mono

    if make_phy_output and not curate_using_phy:
        si.export_to_phy(sorting_analyzer, output_folder=phy_path, remove_if_exists=True, copy_binary=True)
    if make_report:
        si.export_report(sorting_analyzer, output_folder=report_path, remove_if_exists=True)

    # split our extractors back
    cum_rec_samples = np.insert(np.cumsum(rec_samples),0,0)
    sorters = [sorting_analyzer.sorting.frame_slice(start_frame=cum_rec_samples[a], end_frame=cum_rec_samples[a+1] ) for a in range(len(recording_paths))] # get list of sorters

    print(recording_paths)
    # save spike times and waveform information for further analysis
    save_spikes_to_dataframe(sorters, quality_metrics, rec_samples, recording_paths, processed_paths, sorterName)

    return

def make_recording_from_paths_and_get_times(recording_paths):

    rec_times = []
    mono_recording = None
    for recording_path in recording_paths:
        recording_format = get_recording_format(recording_path)
        temp_recording = load_recording(recording_path, recording_format)

        rec_times.append(temp_recording.get_total_samples())
        if mono_recording is None:
            mono_recording = temp_recording
        else:
            mono_recording = si.concatenate_recordings([mono_recording, temp_recording])

    # check if a probe has been added, if not add it
    try: _ = mono_recording.get_probe()
    except: mono_recording, _ = add_probe(mono_recording, recording_paths[0])

    return mono_recording, rec_times
