import sys
from pathlib import Path
import spikeinterface.full as si

from Elrond.Helpers.upload_download import get_chronologized_recording_paths
from Elrond.Helpers.zarr import make_zarrs, delete_zarrs

from Elrond.P1_SpikeSort.spikesort import do_sorting, compute_sorting_analyzer

from Elrond.P1_SpikeSort.defaults import pp_pipelines_dict


mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]

# One of the main tasks in this script is to set up default paths for everything. Default structure is following:
#   project_path/
#       data/M??_D??/
#           M??_D??-date-time-OF1/ raw
#           M??_D??-date-time-OF2/ recordings are
#           M??_D??-date-time-VR/  here
#       derivatives/M??/D??/
#           full/
#               sorter_name/
#                   lotsa stuff: zarrs, sorting_output, sorting_analyzer, report
#           of1/
#               sorter_name/
#                   spikes.pkl
#               dlc/
#               behavioural data
#           of2/
#               sorter_name/
#                   spikes.pkl
#               dlc/
#               behavioural data
#           vr/
#               sorter_name/
#                   spikes.pkl
#               dlc/
#                   licks/
#                   pupil/
#               behavioural data

def do_sorting_pipeline(mouse, day, sorter_name, project_path, pp_for_sorting=None, pp_for_post=None, data_path=None, deriv_path=None, zarr_folder=None, recording_paths=None, sorter_path=None, sa_path=None, report_path=None):
    """
    Do everything related to sorting.
    Start with raw data, end with a sorting_analyzer.
    """

    if data_path is None:
        data_path = project_path + f"data/M{mouse}_D{day}/"
    if recording_paths is None:
        recording_paths = [data_path+"of1/", data_path+"vr/", data_path+"of2/"]

    num_recordings = len(recording_paths)

    if deriv_path is None:
        deriv_path = project_path + f"derivatives/M{mouse}/D{day}/"
    Path(deriv_path).mkdir(exist_ok=True, parents=True)
    if zarr_folder is None:
        zarr_folder = deriv_path + f"full/{sorter_name}/zarr_recordings/"
    zarr_for_sorting_paths = [f"{zarr_folder}/zarr_for_sorting_{a}" for a in range(num_recordings)]

    if pp_for_sorting is None:
        pp_for_sorting = pp_pipelines_dict[sorter_name]["sort"]
    if pp_for_post is None:
        pp_for_post = pp_pipelines_dict[sorter_name]["post"]

    if pp_for_sorting == pp_for_post:
        zarr_for_post_paths = zarr_for_sorting_paths
    else:
        zarr_for_post_paths = [f"{zarr_folder}/zarr_for_post_{a}" for a in range(num_recordings)]
    if sorter_path is None:
        sorter_path = deriv_path + f"full/{sorter_name}/" + sorter_name + "_sorting/"
    if sa_path is None:
        sa_path = deriv_path + f"full/{sorter_name}/" + sorter_name + "_sa"
    if report_path is None:
        report_path = deriv_path + f"full/{sorter_name}/" + sorter_name + "_report/"

    si.set_global_job_kwargs(n_jobs=8)
    make_zarrs(recording_paths, zarr_for_sorting_paths, zarr_for_post_paths, pp_for_sorting, pp_for_post)
    sorting = do_sorting(zarr_for_sorting_paths, sorter_name, sorter_path, deriv_path)
    sorting_analyzer = compute_sorting_analyzer(sorting, zarr_for_post_paths, sa_path)
    si.export_report(sorting_analyzer, report_path)
    delete_zarrs(zarr_for_sorting_paths, zarr_for_post_paths)


raw_recording_paths = get_chronologized_recording_paths(project_path, mouse, day)
do_sorting_pipeline(mouse, day, sorter_name, project_path, recording_paths=raw_recording_paths)
