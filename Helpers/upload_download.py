import os
import shutil
import yaml
import settings
import spikeinterface.full as si
from neuroconv.utils.dict import load_dict_from_file, dict_deep_update

"""
## UNUSED
def save_extractors(sorters, recordings, recording_paths, processed_folder_name,
                    save_recording_extractors=False, save_sorting_extractors=True):
    for sorter, recording, recording_path in zip(sorters, recordings, recording_paths):
        print("I am saving sorter and recording objects for ", recording_path, " at ",
              recording_path + "/" + processed_folder_name)

        if not os.path.exists(recording_path + "/" + processed_folder_name):
            os.mkdir(recording_path + "/" + processed_folder_name)

        if not os.path.exists(recording_path + "/" + processed_folder_name + "/" + settings.sorterName):
            os.mkdir(recording_path + "/" + processed_folder_name + "/" + settings.sorterName)

        sorter_folder = recording_path + "/" + processed_folder_name + "/" + settings.sorterName + "/sorter"
        recording_folder = recording_path + "/" + processed_folder_name + "/recording"

        if os.path.exists(sorter_folder):
            shutil.rmtree(sorter_folder)
        if os.path.exists(recording_folder):
            shutil.rmtree(recording_folder)

        if save_sorting_extractors:
            sorter.save(folder=sorter_folder, n_jobs=4, chunk_size=2000, progress_bar=True, overwrite=True)
        if save_recording_extractors:
            recording.save(folder=recording_folder, n_jobs=4, chunk_size=2000, progress_bar=True, overwrite=True)
"""


def load_recordings(recording_paths, recording_formats):
    recordings = []

    for path, format, in zip(recording_paths, recording_formats):
        if format == "openephys":
            recording = si.read_openephys(path, stream_name='Signals CH')
        elif format == "spikeglx":
            recording = si.read_spikeglx(path)  # untested
        elif format == "nwb":
            recording = si.read_nwb_recording(path)  # untested
        else:
            print("I don't recognise the recording format")
            print("Current options are open_ephys, spikeglx and nwb")
        recordings.append(recording)
    return recordings

def get_recordings_to_postprocess(recording_path, local_path, **kwargs):
    """
    This is a function that returns a list of paths for recordings in which to postprocess.
    This is influenced by concat_sort which is a flag for concatenating recordings before sorting
    """
    recordings_to_sort = [recording_path]
    if ('concat_sort' in kwargs) and ('postprocess_based_on_concat_sort' in kwargs):
        if (kwargs["concat_sort"] == True) and \
                (kwargs["postprocess_based_on_concat_sort"] == True):
            matched_recording_paths = get_matched_recording_paths(recording_path)
            for matched_recording_path in matched_recording_paths:
                matched_recording_name = os.path.basename(matched_recording_path)
                matched_working_recording_path = matched_recording_path
                if local_path in recording_path:
                    matched_working_recording_path = local_path+matched_recording_name
                recordings_to_sort.append(matched_working_recording_path)
                assert os.path.exists(matched_working_recording_path)
    return recordings_to_sort

def get_recordings_to_sort(recording_path, local_path, **kwargs):
    """
    This is a function that returns a list of paths for recordings in which to execute PX_scripts.
    This is influenced by concat_sort which is a flag for concatenating recordings before sorting
    """
    recordings_to_sort = [recording_path]
    if 'concat_sort' in kwargs:
        if kwargs["concat_sort"] == True:
            matched_recording_paths = get_matched_recording_paths(recording_path)
            for matched_recording_path in matched_recording_paths:
                matched_recording_name = os.path.basename(matched_recording_path)
                matched_working_recording_path = matched_recording_path
                if local_path in recording_path:
                    matched_working_recording_path = local_path+matched_recording_name
                recordings_to_sort.append(matched_working_recording_path)
                assert os.path.exists(matched_working_recording_path)
    return recordings_to_sort

def get_recording_types(recording_paths):
    recording_types = []
    for i in range(len(recording_paths)):
        if os.path.exists(recording_paths[i]+"/params.yml"):
            params = load_dict_from_file(recording_paths[i]+"/params.yml")
            if 'recording_type' in params:
                recording_types.append(params['recording_type'])
        else:
            print("I couldn't find a params.yml file for ", recording_paths[i])
            print("I can't assign a recording type without input")
            recording_types.append("NOT A A VALID RECORDING TYPE")
    return recording_types

def get_recording_formats(recording_paths):
    recording_formats = []
    for i in range(len(recording_paths)):
        if os.path.exists(recording_paths[i]+"/params.yml"):
            params = load_dict_from_file(recording_paths[i]+"/params.yml")
            if 'recording_format' in params:
                recording_formats.append(params['recording_format'])
        else:
            print("I couldn't find a params.yml file for ", recording_paths[i])
            print("I will assume it is an open ephys format")
            recording_formats.append("openephys")
    return recording_formats


def copy_folder(src_folder, dest_folder, ignore_items=[]):
    folder_name = os.path.basename(src_folder)

    # add exisiting files to ignore list so they aren't overwritten
    src_folder_files = os.listdir(src_folder)
    dest_folder_files = []
    if os.path.exists(dest_folder):
        dest_folder_files = os.listdir(dest_folder)
    common_files = list(set(src_folder_files) & set(dest_folder_files))
    ignore_items.extend(common_files)

    if folder_name not in ignore_items: # check if in the ignore list
            shutil.copytree(src_folder, dest_folder, ignore=shutil.ignore_patterns(*ignore_items),
                                                     copy_function=copy2_verbose, dirs_exist_ok=True)
    print("")

def copy2_verbose(src, dst):
    print('Copying {0}'.format(src))
    shutil.copy2(src,dst)

def get_matched_recording_paths(recording_path):
    matched_recording_paths = []
    if os.path.isfile(recording_path + "/params.yml"):
        with open(recording_path + "/params.yml", 'r') as f:
            params = yaml.safe_load(f)
            if 'matched_recordings' in params:
                for matched_recording_path in params["matched_recordings"]:
                    matched_recording_paths.append(matched_recording_path)
    return matched_recording_paths

def copy_to_local(recording_path, local_path, **kwargs):
    recordings_to_download = [recording_path]

    # check whether there are more recordings to download locally
    if 'concat_sort' in kwargs:
        if kwargs["concat_sort"] == True:
            matched_recording_paths = get_matched_recording_paths(recording_path)
            recordings_to_download.extend(matched_recording_paths)

    # copy recordings
    for recording_to_download in recordings_to_download:
        recording_name = os.path.basename(recording_to_download)
        if os.path.exists(recording_to_download) and not os.path.exists(local_path+recording_name):
            # results are saved specific to named sorter
            shutil.copytree(recording_to_download,
                            local_path+recording_name, dirs_exist_ok=True)
            print("copied " + recording_to_download + " to " + local_path+recording_name)
        else:
            print("Oh no! Either the recording path or local path couldn't be found")
    return

def copy_from_local(recording_path, local_path, processed_folder_name, **kwargs):
    recordings_to_upload = [recording_path]

    # check whether there are more recordings to download locally
    if 'concat_sort' in kwargs:
        if kwargs["concat_sort"] == True:
            matched_recording_paths = get_matched_recording_paths(recording_path)
            recordings_to_upload.extend(matched_recording_paths)

    # copy recordings
    for recording_to_upload in recordings_to_upload:
        recording_name = os.path.basename(recording_to_upload)
        local_recording_path = local_path + recording_name

        # remove /processed_folder_name from recording on server
        if os.path.exists(recording_path + "/" + processed_folder_name):
            shutil.rmtree(recording_path + "/" + processed_folder_name)

        # copy the processed_folder_name from local to server
        if os.path.exists(local_recording_path + "/" + processed_folder_name):
            shutil.copytree(local_recording_path + "/" + processed_folder_name,
                            recording_to_upload  + "/" + processed_folder_name,
                            copy_function=copy2_verbose, dirs_exist_ok=True)
            print("copied "+local_recording_path+  "/" + processed_folder_name+" to "
                           +recording_to_upload +  "/" + processed_folder_name)
    return

import logging
def _logpath(path, names):
    logging.info('Working in %s' % path)
    return []   # nothing will be ignored

def empty_recording_folder_from_local(local_path):
    for folder_path in os.listdir(local_path):
        if "datastore" in local_path:
            raise Exception("The local path contains 'datastore' in the path name, "
                            "you don't want to delete anything from the datastores server")
        else:
            shutil.rmtree(local_path+folder_path)

def test_upload():
    return

def empty_directory(local_path):
    return

def test_download(recording_path, local_path):
    empty_directory(local_path)
    empty_directory(local_path)
    empty_directory(local_path)
    return

def main():
    # test download and upload
    local_path="/home/ubuntu/to_sort/test_area/"
    recording_path = "/mnt/datastore/Harry/test_recording/vr/M11_D36_2021-06-28_12-04-36"

    test_download(recording_path, local_path)
    test_upload()


if __name__ == '__main__':
    main()