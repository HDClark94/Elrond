import os
import shutil
import yaml
import numpy as np
from pathlib import Path
import spikeinterface.full as si

def get_recording_folders(cohort_folder, mouse, day):
    """
    We (currently) use two folder structures. Either
    data_path/
        of/
            M??_D??_date-time/
                (which might contain openephys, or `recording.zarr` files)
            M??_D??_date-time/
        vr/
            M??_D??_date-time/
            maybe_more/

    or:
    data_path/
        M??_D??/
            a_bunch/
            of_recordings/

    The second follows the BIDS data structure (pretty much)

    This function checks which type we're using, and exports
    the recording folders. It also deals with the case if you
    have a extra /data directory. This is often not used for 
    the raw data but is used when processing.
    """

    recording_folders = None
    data_path = cohort_folder
    if len(list(Path(cohort_folder).glob('data/')))>0:
        data_path += 'data/'

    if len(list(Path(data_path).glob('of/'))) > 0:
        recording_folders = list(Path(cohort_folder + 'of/').glob(f"M{mouse}_D{day}*"))
        recording_folders += list(Path(cohort_folder).glob(f'vr/M{mouse}_D{day}*'))

    elif len(list(Path(data_path).glob(f"*M{mouse}_D{day}"))) > 0:
        recording_folders = list(Path(data_path + f"M{mouse}_D{day}/").glob("*/"))

    for a, recording_folder in enumerate(recording_folders):
        if this_is_zarr(recording_folder) and len(list(Path(recording_folder).rglob('*recording.zarr/'))) > 0:
            recording_folders[a] = list(Path(recording_folder).rglob('*recording.zarr/'))[0]

    for a, recording_folder in enumerate(recording_folders):
        recording_folders[a] = str(recording_folder)

    return recording_folders


def this_is_zarr(recording_folder):
    """
    Checks if a recording_folder is zarr or not.
    Zarr and open_ephys recording are loaded in different ways
    in spikeinterface.
    """

    zarr_recording = False
    if '.zarr' in str(recording_folder) or len(list(Path(recording_folder).rglob('*.zarr/')))>0:
        zarr_recording = True

    return zarr_recording

def get_recording_from(recording_folder):

    if this_is_zarr(recording_folder):
        recording = si.load_extractor(recording_folder)
    else:
        recording = si.read_openephys(recording_folder)

    return recording

def get_raw_recordings_from(recording_paths):
    recordings = []
    for recording_path in recording_paths:
        recordings.append(si.read_openephys(recording_path))
    return recordings

def get_processed_paths(base_processed_path, recording_paths):

    if base_processed_path is None:
        base_processed_path = '/'.join(recording_paths[0].split('/')[:-2]) + '/'

    processed_paths = []
    for recording_path in recording_paths:
        relative_recording_path = '/'.join(recording_path.split('/')[-2:])
        processed_paths.append(base_processed_path + relative_recording_path + '/processed/') 

    return processed_paths

def get_chronologized_recording_paths(data_path, mouse, day):
    recording_paths = get_recording_folders(data_path, mouse, day)
    return chronologize_paths(recording_paths)

def get_recording_paths(data_path, mouse, day):
    """
    Get recording paths based on mouse and day.
    data_path is a path with the recordings one level deeper
    i.e. datapath = "/path/to/recordings/" M1_D1_XXXXXXXX
    """
    mouse_day = "M"+str(mouse)+"_D"+str(day)
    recording_paths = os.listdir(data_path)
    recording_paths = [data_path + s + "/" for s in recording_paths if mouse_day in s]

    return recording_paths

def chronologize_paths(recording_paths):
    """ 
    For a given set of paths, put them in chronological order
    """
    # get basenames of the recordings
    basenames = [os.path.basename(s) for s in recording_paths]
    # split the basename by the first "-" and take only the latter split
    time_dates = [s.split("-", 1)[-1] for s in basenames]
    # reorganise recording_paths based on np.argsort(time_dates)
    recording_paths = np.array(recording_paths)[np.argsort(time_dates)]
    return recording_paths.tolist()


def load_recording(recording_path, recording_format):
    # load recording channels but don't load ADC channels

    if recording_format == "openephys":
        files = [f for f in Path(recording_path).iterdir()]
        if np.any([".continuous" in f.name and f.is_file() for f in files]):
            recording = si.read_openephys(recording_path, stream_name='Signals CH') # format = 'legacy'
        else:
            recording = si.read_openephys(recording_path) # format = 'binary'

    elif recording_format == "spikeglx":
        recording = si.read_spikeglx(recording_path)  # untested
    elif recording_format == "nwb":
        recording = si.read_nwb_recording(recording_path)  # untested
    else:
        raise AssertionError("I don't recognise the recording format,"
                        "Current options are open_ephys, spikeglx and nwb")

    #recording = recording.frame_slice(start_frame=0, end_frame=int(15 * 30000))  # debugging purposes
    return recording


def load_recordings(recording_paths, recording_formats):
    recordings = []
    for path, format, in zip(recording_paths, recording_formats):
        recordings.append(load_recording(path,format))
    return recordings


def get_recording_types(recording_paths):
    recording_types = []
    for i in range(len(recording_paths)):
        if os.path.exists(recording_paths[i]+"/params.yml"):
            params = yaml.safe_load(Path(recording_paths[i]+"/params.yml").read_text())
            if 'recording_type' in params:
                recording_types.append(params['recording_type'])
        else:
            print("I couldn't find a params.yml file for ", recording_paths[i])
            print("I can't assign a recording type without input")
            recording_types.append("NOT A A VALID RECORDING TYPE")
    return recording_types

def get_recording_type(recording_path):
    if os.path.exists(recording_path+"/params.yml"):
        params = yaml.safe_load(Path(recording_path+"/params.yml").read_text())
        if 'recording_type' in params:
            return params['recording_type']
    else:
        print("I couldn't find a params.yml file for ", recording_path)
    return None


def get_recording_format(recording_path):
    if os.path.isfile(recording_path + "/params.yml"):
        with open(recording_path + "/params.yml", 'r') as f:
            params = yaml.safe_load(f)
            if 'recording_format' in params:
                return params["recording_format"]
    print("I could not extract the format from the param.yml")
    return "unknown"


def get_recording_formats(recording_paths):
    recording_formats = []
    for i in range(len(recording_paths)):
        if os.path.exists(recording_paths[i]+"/params.yml"):
            params = yaml.safe_load(Path(recording_paths[i]+"/params.yml").read_text())
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
            shutil.copytree(recording_to_download, local_path+recording_name, dirs_exist_ok=True)
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
        if os.path.exists(recording_to_upload + "/" + processed_folder_name):
            shutil.rmtree(recording_to_upload + "/" + processed_folder_name)

        # copy the processed_folder_name from local to server
        if os.path.exists(local_recording_path + "/" + processed_folder_name):
            shutil.copytree(local_recording_path + "/" + processed_folder_name,
                            recording_to_upload  + "/" + processed_folder_name,
                            copy_function=copy2_verbose)
            print("copied "+local_recording_path+  "/" + processed_folder_name+" to "
                           +recording_to_upload +  "/" + processed_folder_name)
    return


def empty_recording_folder_from_local(path):
    for folder_path in os.listdir(path):
        if "datastore" in path:
            raise Exception("The local path contains 'datastore' in the path name, "
                            "you don't want to delete anything from the datastores server")
        else:
            shutil.rmtree(path+folder_path)


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
