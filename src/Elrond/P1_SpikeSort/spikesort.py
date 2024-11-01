import pandas as pd
import pickle
from Elrond.Helpers.upload_download import *
from .preprocess import preprocess, ammend_preprocessing_parameters
from .auto_curate import auto_curation
from .probe import add_probe 
import Elrond.settings as settings

from Elrond.P1_SpikeSort.defaults import sorter_kwargs_dict, pp_pipelines_dict, default_extensions_dict

from os.path import expanduser
si.set_global_job_kwargs(n_jobs=1)

# this should be in spikeinterface soon
# TODO: when it is, delete
pp_function_to_class = {
    # filter stuff
    "filter": si.FilterRecording,
    "bandpass_filter": si.BandpassFilterRecording,
    "notch_filter": si.NotchFilterRecording,
    "highpass_filter": si.HighpassFilterRecording,
    "gaussian_filter": si.GaussianFilterRecording,
    # gain offset stuff
    "normalize_by_quantile": si.NormalizeByQuantileRecording,
    "scale": si.ScaleRecording,
    "zscore": si.ZScoreRecording,
    "center": si.CenterRecording,
    # decorrelation stuff
    "whiten": si.WhitenRecording,
    # re-reference
    "common_reference": si.CommonReferenceRecording,
    "phase_shift": si.PhaseShiftRecording,
    # misc
    "frame_slice": si.FrameSliceRecording,
    "rectify": si.RectifyRecording,
    "clip": si.ClipRecording,
    "blank_staturation": si.BlankSaturationRecording,
    "silence_periods": si.SilencedPeriodsRecording,
    "remove_artifacts": si.RemoveArtifactsRecording,
    "zero_channel_pad": si.ZeroChannelPaddedRecording,
    "deepinterpolate": si.DeepInterpolatedRecording,
    "resample": si.ResampleRecording,
    "decimate": si.DecimateRecording,
    "highpass_spatial_filter": si.HighpassSpatialFilterRecording,
    "interpolate_bad_channels": si.InterpolateBadChannelsRecording,
    "depth_order": si.DepthOrderRecording,
    "average_across_direction": si.AverageAcrossDirectionRecording,
    "directional_derivative": si.DirectionalDerivativeRecording,
    "astype": si.AstypeRecording,
    "unsigned_to_signed": si.UnsignedToSignedRecording,
}

def apply_pipeline(recording, pp_dict):
       
    pp_recording = recording
    for preprocessor, kwargs in pp_dict.items():
        pp_recording = pp_function_to_class[preprocessor.split(".")[-1]](recording, **kwargs)

    return pp_recording

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
        # new_spike_data = new_spike_data.merge(quality_metrics, on='cluster_id')

        pkl_folder = processed_path + sorterName + "/"
        print(pkl_folder)
        Path(pkl_folder).mkdir(parents=True, exist_ok=True)
        print("I am saving the spike dataframe for ", recording_path, " in ", pkl_folder)
        new_spike_data.to_pickle(pkl_folder + "spikes.pkl")

def save_spikes_per_session(sorting, sorter_name, zarr_for_sorting_paths, deriv_path):

    of1_path, of2_path, vr_path = [deriv_path + ["of1/", "of2/", "vr/"][a] for a in range(3)]
    
    recordings = [si.load_extractor(sorting_path +".zarr") for sorting_path in zarr_for_sorting_paths]
    rec_samples = [recording.get_total_samples() for recording in recordings]
    
    cum_rec_samples = [0]
    for a, rec_sample in enumerate(rec_samples):
        cum_rec_samples.append( cum_rec_samples[a] + rec_sample ) 

    # save spikes, split for sessions
    sorters = [sorting.frame_slice(start_frame=cum_rec_samples[a],
                                   end_frame=cum_rec_samples[a+1] ) for a in
               range(len(rec_samples))] # get list of sorters
    save_spikes_to_dataframe(sorters, None, rec_samples, [deriv_path, deriv_path, deriv_path], [of1_path, vr_path, of2_path], sorter_name)

    return 


def do_sorting(recording_paths, sorter_name, sorter_path, deriv_path, sorter_kwargs=None):
    """
    Does a spike sorting for all paths in `recording_paths`. These recordings should already
    have been preprocessed. The recordings are concatenated together. The spike trains are saved
    for each recording, assuming a 3 recording structure, into of1/, vr/, of2/ folders (based on
    the ... experiment structure). If this is not the strucutre of the experiment, this step
    will (cleanly) fail.
    """

    if sorter_kwargs is None:
        sorter_kwargs = sorter_kwargs_dict[sorter_name]

    # sorting time! We assume the preprocessed recording has been saved as a zarr file.
    # If you're using raw data, use si.read_openephys (or similar) and apply a preprocessor.
    recording_for_sort = si.concatenate_recordings( [
        si.load_extractor(recording_path + ".zarr") for recording_path in recording_paths ] )
    sorting = si.run_sorter_by_property(
            recording=recording_for_sort,
            sorter_name=sorter_name,
            folder=sorter_path,
            remove_existing_folder=True,
            verbose=True, **sorter_kwargs,
            grouping_property='group'
    )
    sorting = si.remove_excess_spikes(recording=recording_for_sort, sorting=sorting)
    
    try:
        save_spikes_per_session(sorting, sorter_name, recording_paths, deriv_path)
    except:
        print("Couldn't save spikes.pkl file for of1, vr, of2 experiment.")
    
    return sorting

def compute_sorting_analyzer(sorting, zarr_for_post_paths, sa_path, extension_dict=None):
    """
    Create a `sorting_analyzer` object, from spikeinterface, which combines the recording
    and sorting together for postprocessing. The extensions in `extension_dict` are
    computed. If None are supplied a computationally minimal set are computed, from the
    Elrond/P1_SpikeSort/defaults.py file"""

    if extension_dict is None:
        extension_dict = default_extensions_dict 

    recording_for_post = si.concatenate_recordings( [
        si.load_extractor(zarr_post+".zarr") for zarr_post in zarr_for_post_paths ] )
    sa = si.create_sorting_analyzer(recording = recording_for_post, 
                                    sorting=sorting, format="binary_folder",
                                folder=sa_path, overwrite=True)
    sa.compute(extension_dict)
    return sa

# TODO: delete!!
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
        automated_model_path = None,
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
            folder=sorting_analyzer_path,
        )
    else:
        sorting_analyzer = si.load_sorting_analyzer(sorting_analyzer_path, load_extensions=True)
        sorting_analyzer._recording = recording_mono

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

        best_pipeline = pickle.load(open(automated_model_path + "best_model_label.pkl","rb"))
        automated_labels = si.auto_label_units(
            sorting_analyzer,
            pipeline=best_pipeline,
            label_conversion = {0: "good", 1: "mua", 2: "noise"}
        )
        save_labels(sorting_analyzer)

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

def save_labels(sorting_analyzer):

    properties_path = sorting_analyzer.folder / 'sorting/properties'

    label_predictions = sorting_analyzer.sorting.get_property('label_prediction')
    label_confidence = sorting_analyzer.sorting.get_property('label_confidence')

    np.save(properties_path / 'label_predictions.npy', np.array(label_predictions))
    np.save(properties_path / 'label_confidence.npy', np.array(label_confidence))

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
