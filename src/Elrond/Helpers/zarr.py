from Elrond.P1_SpikeSort.spikesort import apply_pipeline
from Elrond.Helpers.upload_download import get_raw_recordings_from
import subprocess
import spikeinterface.full as si

def save_one_zarr(rec, zarr_path, preprocessing_pipeline):

    recordings = rec.split_by('group', outputs='list')
    pp_recordings = []

    for group_recording in recordings:

        #bad_channels, _ = si.detect_bad_channels(group_recording)
        #group_recording = group_recording.remove_channels(remove_channel_ids=bad_channels)
        group_recording = apply_pipeline(group_recording, preprocessing_pipeline)
        
        pp_recordings.append(apply_pipeline(group_recording, preprocessing_pipeline))

    pp_recording = si.aggregate_channels(pp_recordings)
    pp_recording.save_to_zarr(zarr_path)

def make_zarrs(recording_paths, zarr_for_sorting_paths, zarr_for_post_paths, pp_for_sorting=None, pp_for_post=None):
    """
    Creates preprocessed zarr files from raw recordings.
    
    We do this as the sorters are _much faster_ at running on already-preprocessed data,
    rather than doing the preprocessing on the fly. It is common to want to change the preprocessing
    for the sorting. It is uncommon to want to change the preprocessing for the postprocessing.

    Parameters
    ----------
    recording_paths : list of str
        List of paths pointing to the raw data
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
        try:
            save_one_zarr(rec, zarr_sorting_path, pp_for_sorting)
        except:
            print(f"Couldn't save {zarr_sorting_path}. Probably exists already.")
        if pp_for_sorting != pp_for_post:
            try:
                save_one_zarr(rec, zarr_post_path, pp_for_post)
            except:
                print(f"Couldn't save {zarr_post_path}. Probably exists already.")

    recordings = None
    return


def delete_zarrs(zarr_for_sorting_paths, zarr_for_post_paths):
    
    for zarr_path in zarr_for_sorting_paths + zarr_for_post_paths:
        if '.zarr' in zarr_path:
            subprocess.run(['rm', '-r', zarr_path])

    return
