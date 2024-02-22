# make sure python-neo is installed in the virtual environment buy navigating within /python-neo and installing with "pip install ."
# Note: this python-neo is a fork of the original that affords can read OpenEphys legacy data without .event films
import os
from neuroconv.utils.dict import load_dict_from_file, dict_deep_update
from neuroconv.datainterfaces import OpenEphysRecordingInterface
import numpy as np

def make_yaml_from_schema(interface, metadata_path, descriptions=True, existing_data=None):
    the_schema = interface.get_metadata_schema()

    metadata_file = open(metadata_path, 'w')

    for level1key in list(the_schema['properties'].keys()):
        print(f"{level1key}:", file=metadata_file)

        level2keys = list(the_schema['properties'][level1key]['properties'].keys())
        if 'definitions' in level2keys:
            level2keys.remove('definitions')

        for level2key in level2keys:
            print(f"  {level2key}: ", end="", file=metadata_file)

            if existing_data is not None and level2key in existing_data[level1key] and type(
                    existing_data[level1key][level2key]) == str:
                print(f"{existing_data[level1key][level2key]}", end="", file=metadata_file)

            if descriptions == True:
                try:
                    print(f" # {the_schema['properties'][level1key]['properties'][level2key]['description']}",
                          file=metadata_file, end="")
                except:
                    if level1key != 'Ecephys':
                        print(f" # No description given", file=metadata_file, end="")

            print("", file=metadata_file)

            if the_schema['properties'][level1key]['properties'][level2key]['type'] == 'array':

                if '$ref' in the_schema['properties'][level1key]['properties'][level2key]['items']:
                    print(f"    - {{\n", end="", file=metadata_file)
                    for level3key in the_schema['properties']['Ecephys']['properties']['definitions'][level2key][
                        'properties'].keys():

                        print(f"    {level3key}: ,", file=metadata_file, end="")
                        # print(f"{level1key}, {level2key}, {level3key}", file=metadata_file, end="")

                        if descriptions == True:
                            try:
                                print(
                                    f" # {the_schema['properties'][level1key]['properties']['definitions'][level2key]['properties'][level3key]['description']}",
                                    file=metadata_file, end="")
                            except:
                                print(f" # No description given", file=metadata_file, end="")
                        print("", file=metadata_file)

                    print(f"}}", file=metadata_file)

                else:
                    print(f"    -", file=metadata_file)

            if level2key == 'ElectricalSeries':
                # print("", file=metadata_file)

                for level3key in list(
                        the_schema['properties'][level1key]['properties'][level2key]['properties'].keys()):
                    print(f"    {level3key}: ", end="", file=metadata_file)

                    if existing_data is not None and level3key in existing_data[level1key][level2key] and type(
                            existing_data[level1key][level2key][level3key]) == str:
                        print(f"{existing_data[level1key][level2key][level3key]}", end="", file=metadata_file)

                    if descriptions == True:
                        try:
                            print(
                                f" # {the_schema['properties'][level1key]['properties'][level2key]['properties'][level3key]['description']}",
                                file=metadata_file, end="")
                        except:
                            print(f" # No description given", file=metadata_file, end="")

                    print("", file=metadata_file)
    metadata_file.close()

def convert(recording_path, processed_folder_name, **kwargs):
    recording_name = os.path.basename(recording_path)

    if not os.path.exists(recording_path + "/nwb/"):
        os.mkdir(recording_path + "/nwb/")

    my_ephys_interface = OpenEphysRecordingInterface(folder_path=recording_path, stream_name="Signals CH")
    existing_data = my_ephys_interface.get_metadata()

    metadata_path = "/mnt/datastore/Harry/basic_metadata.yml" # basic
    nwbfile_path = recording_path+"/"+processed_folder_name+"/nwb/"+recording_name+".nwb"  # Where to save the .nwb file
    #make_yaml_from_schema(my_ephys_interface, metadata_path, descriptions=True, existing_data=existing_data)
    metadata = load_dict_from_file(file_path=metadata_path)
    """
    metadata["NWBFile"]["keywords"] = ["",""]
    metadata["NWBFile"]["related_publications"] = ["", ""]
    metadata["NWBFile"]["experiment_description"] = "unlassified1"
    metadata["NWBFile"]["experimenter"] = ["", ""]
    metadata['NWBFile']['session_id'] = "unclassified4"
    metadata['NWBFile']['institution'] = "unclassified5"
    metadata['NWBFile']['lab'] = "unclassified6"
    metadata['NWBFile']['session_start_time'] = "2000-10-31T01:30:00.000-05:00"
    metadata['NWBFile']['surgery'] = "unclassified7"
    metadata['NWBFile']['pharmacology'] = "unclassified8"
    metadata['NWBFile']['protocol'] = "unclassified9"
    metadata['NWBFile']['surgery'] = "unclassified10"
    metadata['NWBFile']['slices'] = "unclassified11"
    metadata['NWBFile']['source_script'] = "unclassified12"
    metadata['NWBFile']['source_script_file_name'] = "unclassified13"
    metadata['NWBFile']['notes'] = "unclassified14"
    metadata['NWBFile']['virus'] = "unclassified15"
    metadata['NWBFile']['data_collection'] = "unclassified18"
    metadata['NWBFile']['stimulus_notes'] = "unclassified19"
    metadata["Subject"]["subject_id"] = "unclassified3"
    metadata["Subject"]["species"] = "Pachyuromys duprasi"
    metadata["Subject"]["description"] = "unclassified3"
    metadata["Subject"]["genotype"] = "unclassified3"
    metadata["Subject"]["strain"] = "unclassified3"
    metadata["Subject"]["sex"] = "M"
    metadata["Subject"]["weight"] = "unclassified3"
    metadata["Subject"]["date_of_birth"] = "2000-10-31T01:30:00.000-05:00"
    metadata["Subject"]["age"] = "unclassified3"
    metadata["Subject"]["age__reference"] = "birth"
    """
    if isinstance(metadata, dict):
        my_ephys_interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)
    return

def main():
    print("")

if __name__ == '__main__':
    main()
