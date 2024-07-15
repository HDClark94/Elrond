import os
import sys
import traceback
import warnings
import settings
from P1_SpikeSort.spikesort import spikesort, update_from_phy
from P2_PostProcess.postprocess import postprocess

if settings.suppress_warnings:
    warnings.filterwarnings("ignore")

#========================FOR RUNNING ON FROM TERMINAL=====================================#
#=========================================================================================#

recording_path = os.environ['RECORDING_PATH']
local_path = os.environ['LOCAL_PATH']
run_spikesorting = os.environ['SPIKESORT']
update_results_from_phy = os.environ['UPDATE_FROM_PHY']
run_postprocessing = os.environ['POSTPROCESS']
concat_sort = os.environ['CONCATSORT']
use_dlc = os.environ['DLC']
sorterName = os.environ['SORTER']

#=========================================================================================#
#=========================================================================================#

try:
    processed_folder_name = settings.processed_folder_name
    working_recording_path = recording_path  # set as default

    if run_spikesorting:
        spikesort(working_recording_path, local_path, processed_folder_name,
                  sorterName=sorterName, concat_sort=concat_sort)

    if update_results_from_phy:
        update_from_phy(working_recording_path, local_path, processed_folder_name,
                        sorterName=sorterName, concat_sort=concat_sort)

    if run_postprocessing:
        postprocess(working_recording_path, local_path, processed_folder_name,
                    sorterName=sorterName, concat_sort=concat_sort, use_dlc_to_extract_openfield_position=use_dlc)

except Exception as ex:
    print('There was a problem! This is what Python says happened:')
    print(ex)
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback)
    print("")

