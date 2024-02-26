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
def create_param_yml(recording_path, **kwargs):

    print("param.yml created at ", recording_path)
    return


def main():
    print("")
    print("")

if __name__ == '__main__':
    main()
