import os
import yaml
"""
This script works to transcribe deprecated parameter.txt files into params.yml files
Where no values can be found to fill in param.yml, it will be inferred from the recording
e.g. if there are 16 channels, by default it will be inferred as a tetrode recording unless specified otherwise

format of parameter.txt:
----------------------------------------------------------------------------
first line:  recording path on the server
second line: session type used to define which post processing script to run
third line: additional tags seperated by "*"
----------------------------------------------------------------------------

example file looks like this
----------------------------------------------------------------------------
vr
John/ephys_recordings/vr/M1_D1_01-01-12-00-00
paired=John/ephys_recordings/of/M1_D1_01-01-12-00-00*session_type_paired=vr*stop_threshold=4.7*track_length=200
----------------------------------------------------------------------------
"""

def get_tags_parameter_file(recording_directory):
    tags = False
    parameters_path = recording_directory + '/parameters.txt'
    param_file_reader = open(parameters_path, 'r')
    parameters = param_file_reader.readlines()
    parameters = list([x.strip() for x in parameters])
    if len(parameters) > 2:
        tags = parameters[2]
    return tags


def create_param_yml(recording_path, **kwargs):
    if os.path.isfile(recording_path + "/params.yml"):
        print("params.yml already found at" + recording_path + "/params.yml")
    else:
        print("I couldn't find a params.yml at" + recording_path + "/params.yml")
        print("I will try to make one from parameters.txt")
        if os.path.isfile(recording_path+ "/parameters.txt"):
            param_file_reader = open(recording_path+ "/parameters.txt", 'r')
            parameters = param_file_reader.readlines()
            parameters = list([x.strip() for x in parameters])
            recording_type = parameters[0]

            params = dict(
                recording_device="tetrode",
                recording_probe="tetrode",
                n_probes=1,
                recording_aquisition="openephys",
                recording_format="openephys",
                recording_type=recording_type)

            tags = ""
            if len(parameters)>2:
                tags = parameters[2] # a line of tags with "*" splitting each tag
            tags = [x.strip() for x in tags.split('*')] # a list of tags eg. "track_length=200"

            for tag in tags:
                if tag.startswith('stop_threshold'):
                    params["stop_threshold"] = float(tag.split("=")[1])
                elif tag.startswith('track_length'):
                    params["track_length"] = float(tag.split("=")[1])
                elif tag.startswith('paired'):
                    matched_recordings = str(tag.split("=")[1]).split(',')
                    for j in range(len(matched_recordings)):
                        if matched_recordings[j].startswith("Harry"):
                            matched_recordings[j] = "/mnt/datastore/"+matched_recordings[j]
                    params["matched_recordings"] = matched_recordings

            with open(recording_path+'/params.yml', 'w') as outfile:
                yaml.dump(params, outfile, default_flow_style=False, sort_keys=False)
            print("param.yml created at ", recording_path+'/params.yml')

    return


def main():
    print("")
    print("")

if __name__ == '__main__':
    main()
