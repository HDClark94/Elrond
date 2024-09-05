import os
import sys
import traceback
import warnings
import settings as settings

from Helpers.upload_download import copy_from_local, copy_to_local, empty_recording_folder_from_local
from P0_Format.NWB_Formatter.nwb_format import format

def process_recordings(recording_paths, local_path="", processed_folder_name= "", copy_locally=False, run_formatter=False, **kwargs):
    """
    :param recording_paths: list of paths to recordings from which to process
    :param local_path: if copy_locally is true, copy to and from this path
    :param processed_folder_name: name of the folder all the processed results will be returned to
    :param copy_locally: flag whether to download results to the local device before processing further,
    results will be uploaded to origin after processing
    :param run_formatter: flag whether to reformat data (e.g. convert to NWB, )
    :param **kwargs:
        See below

    :Keyword Arguments:
        convert2nwb: flag whether to attempt conversion of the data into nwb format in preprocessing
        convert_ADC_to_VRbehaviour: flag whether to attempt to convert ADCs in vr recordings into behavioural tables
        allow_overwrite_nwb: flag whether to allow the nwb file to be written even if it exists
        metadata_path: path where to find the metadata.yml file in which to make the nwb file with
         (https://www.nwb.org/nwb-neurophysiology/)

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

            if run_formatter:
                print("I will now try to format the recording as requested")
                format(working_recording_path, processed_folder_name, **kwargs)

            if copy_locally:
                print("I will copy the recordingfrom local and remove the recording from local")
                copy_from_local(recording_path, local_path, processed_folder_name, **kwargs)
                empty_recording_folder_from_local(local_path) # clear folder from local

        except Exception as ex:
            print('There was a problem! This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
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
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    #metadata_path = "/mnt/datastore/Harry/cohort6_july2020/basic_metadata.yml"

    #recording_paths = []
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    #metadata_path = "/mnt/datastore/Harry/cohort7_october2020/basic_metadata.yml"

    #recording_paths = []
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])
    metadata_path = "/mnt/datastore/Harry/cohort8_may2021/basic_metadata.yml"

    #recording_paths = []
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort9_february2023/of") if f.is_dir()])
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort9_february2023/vr") if f.is_dir()])
    #metadata_path = "/mnt/datastore/Harry/cohort8_may2021/basic_metadata.yml"

    process_recordings(recording_paths,
                       local_path="/home/ubuntu/to_sort/recordings/",
                       processed_folder_name="processed",
                       copy_locally=False,
                       run_formatter=True,
                       convert2nwb=True,
                       convert_ADC_to_VRbehaviour=False,
                       allow_overwrite_nwb=True,
                       metadata_path=metadata_path)

if __name__ == '__main__':
    main()