import os
import sys
import traceback
import warnings
import settings

from Helpers.upload_download import copy_from_local, copy_to_local, empty_recording_folder_from_local
from P1_SpikeSort.spikesort import spikesort, update_from_phy
from P2_PostProcess.postprocess import postprocess


def process_recordings(recording_paths, local_path="", processed_folder_name="", copy_locally=False,
                       run_spikesorting=False, update_results_from_phy=False, run_postprocessing=False, **kwargs):
    # TODO add checks that the requested flags make sense?
    """
    :param recording_paths: list of paths to recordings from which to process
    :param local_path: if copy_locally is true, copy to and from this local path
    :param processed_folder_name: name of the folder all the processed results will be returned to
    :param copy_locally: flag whether to download results to the local device before processing further,
    results will be uploaded to origin after processing
    :param run_spikesorting: flag whether to spikesort
    :param update_results_from_phy: flag whether to check if a phy folder exists and load it as a sorting extractor
    :param run_postprocessing: flag whether to postprocess (requires spike sorted results)
    # flag false if manually curating
    :param **kwargs:
        See below

    :Keyword Arguments:
        concat_sort: flag whether to look for recordings within the same session and spikesort across
        postprocess_based_on_concat_sort: flag whether to post process potentially multiple recordings
        based on the concat_sort flag and the linked recordings in the param.yl of the original recording
        save2phy: flag whether to export_to_phy after spike sorting (useful for manual curation)
        use_dlc_to_extract_openfield_position: flag whether to ignore any processed output from bonsai and instead use
        deeplabcut to extract openfield position from the raw video
        sorterName: string for a named sorted if spikesorting is called, options include:
        'mountainsort4','klusta','tridesclous','hdsort','ironclust','kilosort',
        'kilosort2', 'spykingcircus','herdingspikes','waveclus'. For each sorter, a different set up might be required
        Refer to https://spikeinterface.readthedocs.io/en/latest/install_sorters.html

    :return: processed recording returned to origin
    """

    for recording_path in recording_paths:
        try:
            recording_name = os.path.basename(recording_path)
            print("I will process recording ", recording_path)

            working_recording_path = recording_path # set as default
            if copy_locally:
                print("I will attempt to copy the recording locally")
                copy_to_local(recording_path, local_path, **kwargs)
                working_recording_path = local_path+recording_name

            #========== spike sorting==============#
            if run_spikesorting:
                print("I will now try to spike sort")
                spikesort(working_recording_path, local_path, processed_folder_name, **kwargs)
            if update_results_from_phy: # or refresh with manually sorted results from phys
                update_from_phy(working_recording_path, local_path, processed_folder_name, **kwargs)
                print("I will try to update the sorted results using the phy folder if it exists")
            #======================================#

            if run_postprocessing:
                print("I will now try to postprocess")
                postprocess(working_recording_path, local_path, processed_folder_name, **kwargs)

            if copy_locally:
                print("I will copy the recording from local and remove the recording from local")
                copy_from_local(recording_path, local_path, processed_folder_name, **kwargs)
                empty_recording_folder_from_local(local_path) # clear folder from local

        except Exception as ex:
            print('There was a problem! This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("")
    return


def main():
    if settings.suppress_warnings:
        warnings.filterwarnings("ignore")

    # take a list of recordings to process
    # e.g. recording_paths = ["/mnt/datastore/Harry/test_recording/vr/M11_D36_2021-06-28_12-04-36"] or
    #      recording_paths = []
    #      recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/test_recording/vr") if f.is_dir()])
    # to grab a whole directory of recordings

    recording_paths = []
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])
    #recording_paths = ["/mnt/datastore/Harry/test_recording/vr/M11_D36_2021-06-28_12-04-36"] # example vr tetrode session with a linked of session
    #recording_paths = ["/mnt/datastore/Harry/test_recording/vr/M18_D1_2023-10-30_12-38-29"] # example vr cambridge p1 probe session with a linked of session (2 x 64 channels)
    #recording_paths = ["/mnt/datastore/Harry/Cohort9_february2023/vr/M16_D1_2023-02-28_17-42-27"]
    #recording_paths =["/mnt/datastore/Harry/test_recording/vr/M0_D0_just_behaviour"]
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort11_april2024/vr") if f.is_dir()])
    #recording_paths = ["/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/CDBS_SIDB_storage/NolanLab/"
    #                   "ActiveProjects/Harry/Cohort11_april2024/vr/M19_D9_2024-05-04_14-07-27"]
    recording_paths = ["/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/groups/CDBS_SIDB_storage/NolanLab/"
                       "ActiveProjects/Harry/Cohort11_april2024/of/M20_D14_2024-05-13_17-40-50_OF2"]
    #recording_paths.extend([f.path for f in os.scandir("/run/user/1000/gvfs/smb-share:server=cmvm.datastore.ed.ac.uk,share=cmvm/sbms/"
    #                        "groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Harry/Cohort11_april2024/vr/") if f.is_dir()])
    #recording_paths = ["/mnt/datastore/Harry/Cohort11_april2024/vr/M20_D5_2024-04-30_16-39-47"]
    process_recordings(recording_paths,
                       local_path="/home/ubuntu/to_sort/recordings/",
                       processed_folder_name="processed",
                       copy_locally=False,
                       run_spikesorting=False,
                       update_results_from_phy=False,
                       run_postprocessing=True,
                       concat_sort=False,
                       postprocess_based_on_concat_sort=False,
                       save2phy=True,
                       use_dlc_to_extract_openfield_position=True,
                       sorterName="mountainsort5")

if __name__ == '__main__':
    main()