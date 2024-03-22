from Helpers.upload_download import *

from P2_PostProcess.Shared import shared as shared
from P2_PostProcess.VirtualReality import vr as vr
from P2_PostProcess.OpenField import of as of
from P2_PostProcess.Opto import opto as opto
from P2_PostProcess.Sleep import sleep as sleep

def postprocess(working_recording_path, local_path, processed_folder_name, **kwargs):
    # process behaviour and spike data based on the recording type

    recording_paths = get_recordings_to_postprocess(working_recording_path, local_path, **kwargs)
    recording_types = get_recording_types(recording_paths)

    for recording_path, type in zip(recording_paths, recording_types):

        if not os.path.exists(recording_path + "/" + processed_folder_name):
            os.mkdir(recording_path + "/" + processed_folder_name)

        shared.process(recording_path, processed_folder_name, **kwargs)
        if type == "vr":
            vr.process(recording_path, processed_folder_name, **kwargs)
        elif type == "openfield":
            of.process(recording_path, processed_folder_name, **kwargs)
        elif type == "opto":
            opto.process(recording_path, processed_folder_name, **kwargs)
        elif type == "sleep":
            sleep.process(recording_path, processed_folder_name, **kwargs)
        else:
            print(type, " isn't a recognised recording "
                        "type in postprocessing ")

    print("Finished post-processing...")