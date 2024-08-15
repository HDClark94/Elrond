import os
import sys
import traceback
import warnings
import Elrond.settings as settings

from pathlib import Path
from os.path import expanduser

from Elrond.Helpers.upload_download import copy_from_local, copy_to_local, \
    empty_recording_folder_from_local, get_recording_paths, get_processed_paths
from Elrond.P1_SpikeSort.spikesort import spikesort
from Elrond.P2_PostProcess.postprocess import postprocess

def main():

    mouse = sys.argv[1]
    day = sys.argv[2]

    if settings.suppress_warnings:
        warnings.filterwarnings("ignore")

    # settings section (could externalise)

    # INPUT paths (set as None or delete if not using)
    home_path = expanduser("~")
    project_path = home_path / Path("Chris/Sorting/cohort11")
    #recording_paths = get_recording_paths(project_path, mouse, day)
    recording_paths = [
        project_path / Path("data/M21_D17/M21_D17_2024-05-17_15-49-55_VR1")
    ]
    automated_model_path = project_path / Path("derivatives/automated_curation_model")
    deeplabcut_of_model_path = project_path / Path("openfield_pose_eddie")

    # OUTPUT paths
    base_processed_path = project_path / Path("derivatives/M"+str(mouse)+"/D"+str(day))
    ephys_path = base_processed_path / Path("ephys_onlyvr")
    sorting_analyzer_path = ephys_path / Path("sorting_analyzer")
    phy_path = ephys_path / Path("phy")
    report_path = ephys_path / Path("report")
    processed_paths = get_processed_paths(base_processed_path, recording_paths)
    for processed_path in processed_paths: Path(processed_path).mkdir(parents=True, exist_ok=True)

    # Options zone
    run_spikesorting=True,
    run_postprocessing=True,

    check_paths_exist(automated_model_path)

    #========== spike sorting==============#
    if run_spikesorting:
        print("I will now try to spike sort")
        print("sorting analyzer at: ", sorting_analyzer_path)
        spikesort(
            recording_paths,
            None,
            do_spike_sorting = True,
            do_spike_postprocessing = True,
            make_report = True,
            make_phy_output = False,
            curate_using_phy = False,
            auto_curate = True,
            sorting_analyzer_path=sorting_analyzer_path,
            phy_path = phy_path,
            report_path = report_path,
            automated_model_path = automated_model_path,
            processed_paths = processed_paths,
            **{
                "sorterName" : "kilosort4",
                "sorter_kwargs" : {'do_CAR': False, 'do_correction': True}
            }
        )
    #======================================#

    if run_postprocessing:
        print("I will now try to postprocess")
        postprocess(
            processed_paths, 
            recording_paths=recording_paths, 
            **{
                "use_dlc_to_extract_openfield_position": True,
                "deeplabcut_of_model_path" : deeplabcut_of_model_path
            }
        )

    return

def check_paths_exist(automated_model_path):
    if Path(automated_model_path).exists() is False:
        raise Exception("Cannot find automated curation model. Current, non-existant, path is ", automated_model_path)
    
if __name__ == '__main__':
    main()
