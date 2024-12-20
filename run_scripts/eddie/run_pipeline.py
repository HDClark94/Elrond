import sys
from pathlib import Path
import pandas as pd
import spikeinterface.full as si

from Elrond.P1_SpikeSort.plotting import plot_simple_np_probe_layout
from Elrond.Helpers.upload_download import get_chronologized_recording_paths
from Elrond.Helpers.zarr import make_zarrs, delete_zarrs

from Elrond.P1_SpikeSort.spikesort import do_sorting, compute_sorting_analyzer

import Elrond.P2_PostProcess.VirtualReality.vr as vr
import Elrond.P2_PostProcess.OpenField.of as of
from Elrond.P2_PostProcess.OpenField.spatial_data import run_dlc_of
from Elrond.P2_PostProcess.VirtualReality.spatial_data import run_dlc_vr
from Elrond.P1_SpikeSort.defaults import pp_pipelines_dict

from Elrond.P2_PostProcess.Shared.theta_phase import compute_channel_theta_phase

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
#                   
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
    plot_simple_np_probe_layout(sorting_analyzer.recording,  deriv_path)
    si.export_report(sorting_analyzer, report_path)
    delete_zarrs(zarr_for_sorting_paths, zarr_for_post_paths)


def do_dlc_pipeline(mouse, day, project_path, dlc_of_model_path=None, dlc_vr_model_path =
                    None, data_path = None, recording_paths=None,
                    of1_save_path=None, of2_save_path=None, vr_save_path=None, do_of1=True, do_of2=True, do_vr=True):
    """
    Do everything related to deeplabcut.
    Start with videos, end with csvs generated by deeplabcut.
    """

    if data_path is None:
        data_path = project_path + f"data/M{mouse}_D{day}/"
    if recording_paths is None:
        recording_paths = [data_path+"of1/", data_path+"vr/", data_path+"of2/"]
    if of1_save_path is None:
        of1_save_path = f"{project_path}derivatives/M{mouse}/D{day}/of1/dlc/"
    if of2_save_path is None:
        of2_save_path = f"{project_path}derivatives/M{mouse}/D{day}/of2/dlc/"
    if vr_save_path is None:
        vr_save_path = f"{project_path}derivatives/M{mouse}/D{day}/of2/vr/"
  
    if dlc_of_model_path is not None:
        if do_of1 is True:
            run_dlc_of(recording_paths[0], of1_save_path, **{"deeplabcut_of_model_path": dlc_of_model_path})
        if do_of2 is True:
            run_dlc_of(recording_paths[2], of2_save_path, **{"deeplabcut_of_model_path": dlc_of_model_path})
    
    if dlc_vr_model_path is not None and do_vr is True:
        run_dlc_vr(recording_paths[1], vr_save_path, **{"deeplabcut_of_model_path": dlc_vr_model_path})

def do_behavioural_postprocessing(mouse, day, sorter_name, project_path, data_path=None,
                      recording_paths=None, deriv_path=None, of1_dlc_folder=None,
                      of2_dlc_folder=None):
    """
    Do everything related to behaviour, including combining spike and behaviour data.
    Start with the output of do_sorting_pipeline and do_dlc_pipeline, end with Figures and scores.
    """

    if data_path is None:
        data_path = project_path + f"data/M{mouse}_D{day}/"
    if recording_paths is None:
        recording_paths = [data_path+"of1/", data_path+"vr/", data_path+"of2/"]
    if deriv_path is None:
        deriv_path = f"{project_path}derivatives/M{mouse}/D{day}/"
    if of1_dlc_folder is None:
        of1_dlc_folder = deriv_path + "of1/dlc/"
    if of2_dlc_folder is None:
        of2_dlc_folder = deriv_path + "of2/dlc/"

    for recording_path in recording_paths:
        end_of_name = recording_path.split("_")[-1]
        if end_of_name == 'OF1':
            of1_dlc_csv_path = list(Path(of1_dlc_folder).glob("*200.csv"))[0]
            of1_dlc_data = pd.read_csv(of1_dlc_csv_path, header=[1, 2], index_col=0) 
            of.process(recording_path, deriv_path + "of1/" , of1_dlc_data, **{"sorterName": sorter_name})
        elif end_of_name == 'OF2':
            of2_dlc_csv_path = list(Path(of2_dlc_folder).glob("*200.csv"))[0]
            of2_dlc_data = pd.read_csv(of2_dlc_csv_path, header=[1, 2], index_col=0) 
            of.process(recording_path, deriv_path + "of2/" , of2_dlc_data, **{"sorterName": sorter_name})
        elif end_of_name == 'VR':
            vr.process(recording_path, deriv_path + "vr/", **{"sorterName": sorter_name})
        elif end_of_name == 'MCVR1':
            print('I need to process the visual something whatnot')

        return

def do_theta_phase(mouse, day, project_path, recording_paths):

    deriv_path = project_path + f"derivatives/M{mouse}/D{day}/"
    save_paths = [deriv_path + session for session in ["of1/", "vr/", "of2/"]]

    for recording_path, save_path in zip(recording_paths, save_paths):
        compute_channel_theta_phase(recording_path, save_path)

    return


if __name__ == "__main__":

    mouse = sys.argv[1]
    day = sys.argv[2]
    sorter_name = sys.argv[3]
    project_path = sys.argv[4]

    raw_recording_paths = get_chronologized_recording_paths(project_path, mouse, day)
    do_sorting_pipeline(mouse, day, sorter_name, project_path, recording_paths = raw_recording_paths)
    do_dlc_pipeline(mouse, day, project_path, dlc_of_model_path = project_path + "derivatives/dlc_models/of_cohort12-krs-2024-10-30/", recording_paths = raw_recording_paths)
    do_behavioural_postprocessing(mouse, day, sorter_name, project_path, recording_paths = raw_recording_paths)
