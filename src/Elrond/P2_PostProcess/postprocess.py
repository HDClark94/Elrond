from ..Helpers.upload_download import *

#from P2_PostProcess.Shared import shared as shared
from .VirtualReality import vr as vr
from .OpenField import of as of
from .Opto import opto as opto
from .Sleep import sleep as sleep
from .ABOVisualCoding import visual_coding

from datetime import datetime

def postprocess(processed_folder_name, processed_paths, recording_paths, **kwargs):
    # process behaviour and spike data based on the recording type

    recording_types = get_recording_types(recording_paths)

    for recording_path, type, processed_path in zip(recording_paths, recording_types, processed_paths):

        if not os.path.exists(recording_path + "/" + processed_folder_name):
            os.mkdir(recording_path + "/" + processed_folder_name)


        if type == "vr":
            print("Before vr, time is", datetime.now())
            vr.process(recording_path, processed_path,  **kwargs)
            print("After vr, time is", datetime.now())
        elif type == "openfield":
            print("Before of, time is", datetime.now())
            of.process(recording_path, processed_path, **kwargs)
            print("After of, time is", datetime.now())
        elif type == "opto":
            opto.process(recording_path, processed_path, **kwargs)
        elif type == "sleep":
            sleep.process(recording_path, processed_path, **kwargs)
        elif type == "allen_brain_observatory_visual_coding":
            visual_coding.process(recording_path, processed_path, **kwargs)

        else:
            print(type, " isn't a recognised recording type in postprocessing")
        #shared.process(recording_path, processed_folder_name, **kwargs)

    print("Finished post-processing... and time is ", datetime.now())
