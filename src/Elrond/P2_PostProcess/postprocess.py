from Elrond.Helpers.upload_download import *

#from P2_PostProcess.Shared import shared as shared
from .VirtualReality import vr as vr
from .VirtualRealityMultiContext import vrmc as vrmc
from .OpenField import of as of
from .Opto import opto as opto
from .Sleep import sleep as sleep
from .ABOVisualCoding import visual_coding
from .DVDWaitScreen import dvd as dvd 
from .OpenField.spatial_data import run_dlc_of
from .VirtualReality.spatial_data import run_dlc_vr


def postprocess(processed_folder_name, processed_paths, recording_paths, **kwargs):
    # process behaviour and spike data based on the recording type

    recording_types = get_recording_types(recording_paths)

    for recording_path, type, processed_path in zip(recording_paths, recording_types, processed_paths):
        if type == "vr":
            #run_dlc_vr(recording_path, save_path = processed_path+"video/", **kwargs)
            vr.process(recording_path, processed_path,  **kwargs)
        elif type == "vr_multi_context": 
            vrmc.process(recording_path, processed_path,  **kwargs)
        elif type == "dvd":
            dvd.process(recording_path, processed_path,  **kwargs)
        elif type == "openfield": 
            dlc_position_data = run_dlc_of(recording_path, save_path = processed_path+"video/", **kwargs)
            of.process(recording_path, processed_path, dlc_position_data, **kwargs)
        elif type == "opto":
            opto.process(recording_path, processed_path, **kwargs)
        elif type == "sleep":
            sleep.process(recording_path, processed_path, **kwargs)
        elif type == "allen_brain_observatory_visual_coding":
            visual_coding.process(recording_path, processed_path, **kwargs)
        else:
            print(type, " isn't a recognised recording type in postprocessing")
        #shared.process(recording_path, processed_folder_name, **kwargs)

    print("Finished post-processing...")
