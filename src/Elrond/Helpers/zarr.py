from Elrond.P1_SpikeSort.spikesort import apply_pipeline
from Elrond.Helpers.upload_download import get_raw_recordings_from

def save_one_zarr(rec, zarr_path, preprocessing_pipeline):
    if preprocessing_pipeline == {}:
        rec.save_to_zarr(zarr_path)
    else:
        apply_pipeline(rec, preprocessing_pipeline).save_to_zarr(zarr_path)


def make_zarrs(recording_paths, sorter_name, zarr_for_sorting_paths, zarr_for_post_paths, pp_for_sorting=None, pp_for_post=None):
    """
    Creates preprocessed zarr files from raw recordings.
    
    We do this as the sorters are _much faster_ at running on already-preprocessed data,
    rather than doing the preprocessing on the fly. It is common to want to change the preprocessing
    for the sorting. It is uncommon to want to change the preprocessing for the postprocessing.

    Parameters
    ----------
    recording_paths : list of str
        List of paths pointing to the raw data
    sorter_name : str
        Sorter being used, e.g. 'kilosort4' or 'herdingspikes'
    zarr_for_sorting_paths : list of str
        List of paths of where to saw the preprocessed data for sorting
    zarr_for_post_paths : list of str
        List of paths of where to saw the preprocessed data for postprocessing
    pp_for_sorting : dict or None
        Dict of preprocessing steps for sorting in the form {'preprocess_step': {'kwarg': 'value}}
        E.g. {"common_reference": {"operator": "average"}, "highpass_filter": {}}
        If None, applies default preprocessing for sorter from 'P1_SpikeSort/defaults.py'
    pp_for_postprocessing : dict or None
        Dict of preprocessing steps for postprocessing in the form {'preprocess_step': {'kwarg': 'value}}
        E.g. {"common_reference": {"operator": "average"}, "highpass_filter": {}}
        If None, applies default preprocessing for sorter from 'P1_SpikeSort/defaults.py'
    """

    recordings = get_raw_recordings_from(recording_paths)
    for rec, zarr_sorting_path, zarr_post_path in zip(recordings, zarr_for_sorting_paths, zarr_for_post_paths):
        save_one_zarr(rec, zarr_sorting_path, pp_for_sorting)
        if pp_for_sorting != pp_for_post:
            save_one_zarr(rec, zarr_post_path, pp_for_post)

    recordings = None
    return
