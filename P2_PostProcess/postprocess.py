from Helpers.upload_download import *

def postprocess(working_recording_path, local_path, **kwargs):
    recording_paths = get_recordings_to_postprocess(working_recording_path, local_path, **kwargs)
    recording_types = get_recording_types(recording_paths)

    for recording_path, type in zip(recording_paths, recording_types):
        if type == "vr":
            print("do something")
            # do stuff to do with vr recordings
        elif type == "openfield":
            print("do something")
            # do stuff to do with open field recordings
        elif type == "opto":
            print("do something")
            # do stuff to do with opto recordings
        elif type == "sleep":
            print("do something")
            # do stuff to do with sleep recordings
        else:
            print(type, " isn't a recognised recording "
                        "type in postprocessing ")

    # then do stuff
    return