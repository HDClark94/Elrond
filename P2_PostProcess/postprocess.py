from Helpers.upload_download import *

def postprocess(working_recording_path, local_path, **kwargs):
    recording_paths = get_recordings_to_postprocess(working_recording_path, local_path, **kwargs)

    # then do stuff
    return