import os
from P0_Format.NWB_Formatter.OpenEphys import ConvertOpenEphys2NWB
from P2_PostProcess.VirtualReality.behaviour_from_ADC_channels import generate_position_data_from_ADC_channels, \
    run_checks_for_position_data
from Helpers.upload_download import get_recording_format

def check_kwargs_are_compatible(**kwargs):
    return True

def format(recording_path, processed_folder_name, **kwargs):
    kwargs_compatible = check_kwargs_are_compatible(**kwargs)

    # create the processed folder if not already made
    if not os.path.exists(recording_path + "/" + processed_folder_name):
        os.mkdir(recording_path + "/" + processed_folder_name)

    if "convert_ADC_to_VRbehaviour" in kwargs:
        if kwargs["convert_ADC_to_VRbehaviour"] == True:
            position_data = generate_position_data_from_ADC_channels(recording_path, processed_folder_name)
            run_checks_for_position_data(position_data, recording_path, processed_folder_name)

    # look for flag to convert recordings to nwb format
    if 'convert2nwb' in kwargs:
        if kwargs["convert2nwb"] == True:
            print("I will attempt to convert this file to NWB format")
            recording_format = get_recording_format(recording_path)
            if recording_format == "openephys":
                print("This recording is in openephys format")
                ConvertOpenEphys2NWB.convert(recording_path, processed_folder_name, **kwargs)
            elif recording_format == "spikeglx":
                print("This isn't implemented but could be")
            else:
                print("The conversion to nwb from this format is not implemented")
    return
