from Elrond.P1_SpikeSort.spikesort import apply_pipeline
from Elrond.Helpers.upload_download import get_raw_recordings_from


def save_one_zarr(rec, zarr_path, preprocessing_pipeline):
    if preprocessing_pipeline == {}:
        rec.save_to_zarr(zarr_path)
    else:
        apply_pipeline(rec, preprocessing_pipeline).save_to_zarr(zarr_path)


def make_zarrs(recording_paths, zarr_for_sorting_paths, zarr_for_post_paths, pp_for_sorting, pp_for_post):

    recordings = get_raw_recordings_from(recording_paths)
    for rec, zarr_sorting_path, zarr_post_path in zip(recordings, zarr_for_sorting_paths, zarr_for_post_paths):
        save_one_zarr(rec, zarr_sorting_path, pp_for_sorting)
        if pp_for_sorting != pp_for_post:
            save_one_zarr(rec, zarr_post_path, pp_for_post)

    recordings = None
    return
