from Elrond.P1_SpikeSort.defaults import sorter_kwargs_dict, pp_pipelines_dict
from pathlib import Path
import spikeinterface.full as si
import sys

from Elrond.P1_SpikeSort.defaults import pp_pipelines_dict

from Elrond.Helpers.zarr import make_zarrs
from Elrond.P1_SpikeSort.spikesort import do_sorting, do_postprocessing

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]

def do_sorting_pipeline(mouse, day, sorter_name, project_path, pp_for_sorting=None, pp_for_post=None, data_path=None, deriv_path=None, zarr_folder=None ,
              recording_paths=None, sorter_path=None, sa_path=None, report_path=None):

    # setting up default paths for everything. Default structure is following:
    #   project_path/
    #       data/M??_D??/
    #           of1/ zarr formatted
    #           of2/ recordings are
    #           vr/  here
    #       derivatives/M??/D??/
    #           full/
    #               sorter_name/
    #                   lotsa stuff
    #           of1/
    #               sorter_name/
    #                   spikes.pkl
    #           of2/
    #               sorter_name/
    #                   spikes.pkl
    #           vr/
    #               sorter_name/
    #                   spikes.pkl

    # If your, e.g. recordings, are somewhere else, just pass
    # recording_paths = [path/to/rec1, path/to/rec2] etc.

    if pp_for_sorting is None:
        pp_for_sorting = pp_pipelines_dict[sorter_name]["sort"] 
    if pp_for_post is None:
        pp_for_post = pp_pipelines_dict[sorter_name]["post"]
    if data_path is None:
        data_path = project_path + f"data/M{mouse}_D{day}/"
    if deriv_path is None:
        deriv_path = project_path + f"derivatives/M{mouse}/D{day}/"
    Path(deriv_path).mkdir(exist_ok=True, parents=True)
    if zarr_folder is None:
        zarr_folder = deriv_path + "full/zarr_recordings/"
    zarr_for_sorting_paths = [f"{zarr_folder}/zarr_for_sorting_{a}" for a in range(3)]
    zarr_for_post_paths = [f"{zarr_folder}/zarr_for_post_{a}" for a in range(3)]
    if pp_for_sorting == pp_for_post:
        zarr_for_post_paths = zarr_for_sorting_paths

    if recording_paths is None:
        recording_paths = [data_path+"of1/", data_path+"vr/", data_path+"of2/"]
    if sorter_path is None: 
        sorter_path = deriv_path + "full/" + sorter_name + "_sorting/"
    if sa_path is None:
        sa_path = deriv_path + "full/" + sorter_name + "_sa"
    if report_path is None:
        report_path = deriv_path + "full/" + sorter_name + "_report/"


    si.set_global_job_kwargs(n_jobs=4)
    make_zarrs(recording_paths, zarr_for_sorting_paths, zarr_for_post_paths, pp_for_sorting, pp_for_post)
    sorting = do_sorting(zarr_for_sorting_paths, sorter_name, sorter_path, deriv_path)
    sorting_analyzer = do_postprocessing(sorting, zarr_for_post_paths, sa_path)
    si.export_report(sorting_analyzer, report_path)

do_sorting_pipeline(mouse, day, sorter_name, project_path)


