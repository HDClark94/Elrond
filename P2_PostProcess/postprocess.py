from Helpers.upload_download import *

#from P2_PostProcess.Shared import shared as shared
from P2_PostProcess.VirtualReality import vr as vr
from P2_PostProcess.VirtualRealityMultiContext import vrmc as vrmc
from P2_PostProcess.OpenField import of as of
from P2_PostProcess.Opto import opto as opto
from P2_PostProcess.Sleep import sleep as sleep
from P2_PostProcess.ABOVisualCoding import visual_coding

def postprocess(processed_folder_name, processed_paths, recording_paths, **kwargs):
    # process behaviour and spike data based on the recording type

    recording_types = get_recording_types(recording_paths)

    for recording_path, type, processed_path in zip(recording_paths, recording_types, processed_paths):
        if type == "vr":  
            vr.process(recording_path, processed_path,  **kwargs)
        elif type == "vr_multi_context":
            vrmc.process(recording_path, processed_path,  **kwargs)
        elif type == "openfield":
            of.process(recording_path, processed_path, **kwargs)
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
