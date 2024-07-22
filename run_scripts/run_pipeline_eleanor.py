import os
import sys
import traceback
import warnings
import settings

from pathlib import Path
from os.path import expanduser

from Helpers.upload_download import copy_from_local, copy_to_local, \
    empty_recording_folder_from_local, get_processed_paths
from P1_SpikeSort.spikesort import spikesort
from P2_PostProcess.postprocess import postprocess


def process_recordings(recording_paths,
                       local_path="",
                       processed_folder_name="",
                       copy_locally=False,
                       run_spikesorting=False,
                       run_postprocessing=False,
                       sorting_analyzer_path=None,
                       phy_path=None,
                       report_path=None,
                       base_processed_path=None,
                       **kwargs):
    """
    :param recording_paths: list of paths to recordings from which to process
    :param local_path: if copy_locally is true, copy to and from this local path
    :param processed_folder_name: name of the folder all the processed results will be returned to
    :param copy_locally: flag whether to download results to the local device before processing further,
    results will be uploaded to origin after processing
    :param run_spikesorting: flag whether to spikesort
    :param run_postprocessing: flag whether to postprocess (requires spike sorted results)
    # flag false if manually curating
    :param **kwargs:
        See below

    :Keyword Arguments:
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

    processed_paths = get_processed_paths(base_processed_path, recording_paths)
    for processed_path in processed_paths: Path(processed_path).mkdir(parents=True, exist_ok=True)
    print(recording_paths, processed_paths)
    #========== spike sorting==============#
    if run_spikesorting:
        print("I will now try to spike sort")
        print("sorting analyzer at: ", sorting_analyzer_path)
        spikesort(
            recording_paths,
            local_path,
            processed_folder_name,
            do_spike_sorting = True,
            do_spike_postprocessing = True,
            make_report = True,
            make_phy_output = False,
            curate_using_phy = False,
            auto_curate = False,
            sorting_analyzer_path=sorting_analyzer_path,
            phy_path = phy_path,
            report_path = report_path,
            processed_paths = processed_paths,
            **kwargs
        )
    #======================================#

    if run_postprocessing:
        print("I will now try to postprocess")
        postprocess(working_recording_path, local_path, processed_folder_name, processed_paths, recording_paths=recording_paths, **kwargs)

    for recording_path in recording_paths:
        try:
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


def get_recording_paths(project_path, mouse, day):
    """
    Get recording paths based on mouse and day.
    """
    data_path = project_path + "data/M"+str(mouse)+"_D"+str(day)+"/"
    recording_paths = [ data_path + "of/" + os.listdir(data_path + "of/")[a] for a in range(0,2)]
    recording_paths.append(data_path + "vr/" + os.listdir(data_path + "vr/")[0])

    print(recording_paths)
    return recording_paths

def main():
    if settings.suppress_warnings:
        warnings.filterwarnings("ignore")

    mouse = 20
    day = 14
    mouse_day = "M"+str(mouse)+"_D"+str(day)

    project_path = "/mnt/datastore/Harry/Cohort11_april2024/"
    recording_paths = []
    recording_paths.extend([f.path for f in os.scandir(project_path+"vr") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir(project_path+"of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir(project_path+"allen_brain_observatory_visual_coding") if f.is_dir()])
    recording_paths = [s for s in recording_paths if mouse_day in s]
    ephys_path = project_path + "derivatives/M"+str(mouse)+"/D"+str(day)+"/ephys/"


    process_recordings(
        recording_paths,
        local_path="/home/ubuntu/to_sort/recordings/",
        processed_folder_name="processed/",
        copy_locally=False,
        run_spikesorting=True,
        run_postprocessing=True,
        sorting_analyzer_path= ephys_path + "sorting_analyzer/",
        phy_path = ephys_path + "phy/",
        report_path = ephys_path + "report/",
        base_processed_path = project_path + "derivatives/M"+str(mouse)+"/D"+str(day)+"/",
        sorterName="kilosort4",
        sorter_kwargs={'do_CAR': False, 'do_correction': True}
    )

if __name__ == '__main__':
    main()
