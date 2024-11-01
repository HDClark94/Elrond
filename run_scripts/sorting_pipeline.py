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
    """
    Do everything related to sorting.
    Start with raw data, end with a sorting_analyzer.
    """

    if data_path is None:
        data_path = project_path + f"data/M{mouse}_D{day}/"
    if deriv_path is None:
        deriv_path = project_path + f"derivatives/M{mouse}/D{day}/"
    Path(deriv_path).mkdir(exist_ok=True, parents=True)
    if zarr_folder is None:
        zarr_folder = deriv_path + f"full/{sorter_name}/zarr_recordings/"
    zarr_for_sorting_paths = [f"{zarr_folder}/zarr_for_sorting_{a}" for a in range(3)]
    zarr_for_post_paths = [f"{zarr_folder}/zarr_for_post_{a}" for a in range(3)]

    if recording_paths is None:
        recording_paths = [data_path+"of1/", data_path+"vr/", data_path+"of2/"]
    if sorter_path is None: 
        sorter_path = deriv_path + f"full/{sorter_name}/" + sorter_name + "_sorting/"
    if sa_path is None:
        sa_path = deriv_path + f"full/{sorter_name}/" + sorter_name + "_sa"
    if report_path is None:
        report_path = deriv_path + f"full/{sorter_name}/" + sorter_name + "_report/"


    si.set_global_job_kwargs(n_jobs=8)
    make_zarrs(recording_paths, sorter_name, zarr_for_sorting_paths, zarr_for_post_paths, pp_for_sorting, pp_for_post)
    sorting = do_sorting(zarr_for_sorting_paths, sorter_name, sorter_path, deriv_path)
    sorting_analyzer = compute_sorting_analyzer(sorting, zarr_for_post_paths, sa_path)
    si.export_report(sorting_analyzer, report_path)

raw_recording_paths = get_chronologized_recording_paths(project_path, mouse, day)
do_sorting_pipeline(mouse, day, sorter_name, project_path, recording_paths = raw_recording_paths)

