import sys
import os
from pathlib import Path
import pandas as pd

import spikeinterface.full as si

from Elrond.Helpers.upload_download import get_chronologized_recording_paths
from Elrond.Helpers.zarr import make_zarrs

from Elrond.P1_SpikeSort.spikesort import do_sorting, compute_sorting_analyzer

import Elrond.P2_PostProcess.VirtualReality.vr as vr
import Elrond.P2_PostProcess.OpenField.of as of
from Elrond.P2_PostProcess.OpenField.spatial_data import run_dlc_of
from Elrond.P2_PostProcess.VirtualReality.spatial_data import run_dlc_vr


mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]

def do_sorting_pipeline(mouse, day, sorter_name, project_path, pp_for_sorting=None, pp_for_post=None, data_path=None, deriv_path=None, zarr_folder=None ,
              recording_paths=None, sorter_path=None, sa_path=None, report_path=None):

    # setting up default paths for everything. Default structure is following:
    #   project_path/
    #       data/M??_D??/
    #           of1/ raw
    #           of2/ recordings are
    #           vr/  here
    #       derivatives/M??/D??/
    #           full/
    #               sorter_name/
    #                   lotsa stuff: zarrs, sorting_output, sorting_analyzer, report
    #           of1/
    #               sorter_name/
    #                   spikes.pkl
    #           of2/
    #               sorter_name/
    #                   spikes.pkl
    #           vr/
    #               sorter_name/
    #                   spikes.pkl
    #   
    # If your, e.g. recordings, are somewhere else, just pass
    # recording_paths = [path/to/rec1, path/to/rec2] etc.


    if data_path is None:
        data_path = project_path + f"data/M{mouse}_D{day}/"
    if deriv_path is None:
        deriv_path = project_path + f"derivatives/M{mouse}/D{day}/"
    Path(deriv_path).mkdir(exist_ok=True, parents=True)
    if zarr_folder is None:
        zarr_folder = deriv_path + "full/{sorter_name}/zarr_recordings/"
    zarr_for_sorting_paths = [f"{zarr_folder}/zarr_for_sorting_{a}" for a in range(3)]
    zarr_for_post_paths = [f"{zarr_folder}/zarr_for_post_{a}" for a in range(3)]

    if recording_paths is None:
        recording_paths = [data_path+"of1/", data_path+"vr/", data_path+"of2/"]
    if sorter_path is None: 
        sorter_path = deriv_path + "full/{sorter_name}/" + sorter_name + "_sorting/"
    if sa_path is None:
        sa_path = deriv_path + "full/{sorter_name}/" + sorter_name + "_sa"
    if report_path is None:
        report_path = deriv_path + "full/{sorter_name}/" + sorter_name + "_report/"


    si.set_global_job_kwargs(n_jobs=4)
    make_zarrs(recording_paths, zarr_for_sorting_paths, zarr_for_post_paths, pp_for_sorting, pp_for_post)
    sorting = do_sorting(zarr_for_sorting_paths, sorter_name, sorter_path, deriv_path)
    sorting_analyzer = compute_sorting_analyzer(sorting, zarr_for_post_paths, sa_path)
    si.export_report(sorting_analyzer, report_path)

def do_dlc_pipeline(mouse, day, dlc_of_model_path=None, dlc_vr_model_path =
                    None, data_path = None, recording_paths=None,
                    of1_save_path=None, of2_save_path=None, vr_save_path=None):

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
  
    of1_path, vr_path, of2_path = [data_path + "of1/", data_path + "vr/",
                                   data_path + "of2/"]

    if dlc_of_model_path is not None:
        run_dlc_of(of1_path, of1_save_path, **{"deeplabcut_of_model_path": dlc_of_model_path})
        run_dlc_of(of2_path, of2_save_path, **{"deeplabcut_of_model_path": dlc_of_model_path})
    
    if dlc_vr_model_path is not None: 
        run_dlc_vr(vr_path, vr_save_path, **{"deeplabcut_of_model_path": dlc_vr_model_path})

def do_postprocessing(mouse, day, sorter_name, project_path, data_path=None,
                      recording_paths=None, deriv_path=None, of1_dlc_folder=None,
                      of2_dlc_folder=None):

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

    # vr
    vr.process(recording_paths[1], deriv_path + "vr/", **{"sorterName":
                                                          sorter_name})

    # of1
    of1_dlc_csv_path = list(Path(of1_dlc_folder).glob("*200.csv"))[0]
    of1_dlc_data = pd.read_csv(of1_dlc_csv_path, header=[1, 2], index_col=0) 
    of.process(recording_paths[0], deriv_path + "of1/" , of1_dlc_data, **{"sorterName": sorter_name})

    # of2
    of2_dlc_csv_path = list(Path(of2_dlc_folder).glob("*200.csv"))[0]
    of2_dlc_data = pd.read_csv(of2_dlc_csv_path, header=[1, 2], index_col=0) 
    of.process(recording_paths[2], deriv_path + "of2/" , of2_dlc_data, **{"sorterName": sorter_name})

raw_recording_paths = get_chronologized_recording_paths(project_path + "data/", mouse, day)
do_sorting_pipeline(mouse, day, sorter_name, project_path, recording_paths = raw_recording_paths)
do_dlc_pipeline(mouse, day, dlc_of_model_path = project_path + "derivatives/dlc_models/of_cohort12-krs-2024-10-30/")
do_postprocessing(mouse, day, sorter_name, project_path)






