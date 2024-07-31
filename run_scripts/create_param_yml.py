import os
import sys
import traceback
import warnings
import Elrond.settings as settings
import yaml
import pandas as pd
import numpy as np

def get_subject(recording_name):
    return recording_name.split("_")[0]


def get_day(recording_name):
    tmp = recording_name.split("_")[1]
    tmp = tmp.split("D")[1]
    tmp = ''.join(filter(str.isdigit, tmp))
    return int(tmp)


def get_recording_type(recording):
    sub_folder = recording.split("/")[-2]
    if sub_folder == "of" or sub_folder == "OpenFeild" or sub_folder == "OpenField":
        return "openfield"
    elif sub_folder == "vr" or sub_folder == "VirtualReality":
        return "vr"
    elif sub_folder == "allen_brain_visual_coding" or sub_folder == "allen_brain_observatory_visual_coding":
        return "allen_brain_observatory_visual_coding"
    else:
        raise AssertionError("Subfolder for the recording should be named in accordance to the recording type")


def create_param_yml(recording_path, recording_paths, parameter_helper_path=None, allow_overwrite=False):
    if not os.path.isfile(recording_path + "/params.yml"):
        print("no params.yml found at" + recording_path + "/params.yml")
        create = True
    else:
        print("params.yml already found" + recording_path + "/params.yml")
        create = False
    if allow_overwrite:
        create = True

    if create:
        recording_name = os.path.basename(recording_path)
        mouse_id = get_subject(recording_name)
        day = get_day(recording_name)
        recording_type = get_recording_type(recording_path)

        if parameter_helper_path is not None:
            print("I will try to make one from a parameter_helper file")
            if os.path.isfile(parameter_helper_path):
                helper = pd.read_csv(parameter_helper_path)
                helper_mouse = helper[helper["mouse_id"] == mouse_id]
                helper_day = helper_mouse[helper_mouse["training_day"] == day]

                # basic set of parameters to define the recording
                params = dict(
                    recording_device="",
                    recording_probe="",
                    recording_aquisition="",
                    recording_format="",
                    recording_type=recording_type)

                for column, dtype in zip(list(helper_day), helper_day.dtypes):
                    if dtype == object:
                        params[column] = str(helper_day[column].iloc[0])
                    if dtype == np.int64:
                        params[column] = int(helper_day[column].iloc[0])
                    if dtype == np.float64:
                        params[column] = float(helper_day[column].iloc[0])

                # load matched recordings
                matched_recordings = []
                for other_recording_path in recording_paths:
                    other_recording_name = os.path.basename(other_recording_path)
                    other_mouse_id = get_subject(other_recording_name)
                    other_day = get_day(other_recording_name)
                    if (other_recording_name != recording_name) & \
                       (other_day == day) & \
                       (other_mouse_id == mouse_id):
                        matched_recordings.append(other_recording_path)
                params["matched_recordings"] = matched_recordings

            with open(recording_path+'/params.yml', 'w') as outfile:
                yaml.dump(params, outfile, default_flow_style=False, sort_keys=False)
            print("param.yml created at ", recording_path+'/params.yml')
        else:
            raise AssertionError("I need a parameter helper file to make a param.yml")
    return

def process_recordings(recording_paths, parameter_helper_path="", allow_overwrite=False):
    """
    :param recording_paths: list of paths to recordings from which to process
    :return:
    """
    for recording_path in recording_paths:
        try:
            print("I will process recording ", recording_path)
            create_param_yml(recording_path, recording_paths, parameter_helper_path, allow_overwrite)

        except Exception as ex:
            print('There was a problem! This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
    return


def main():
    if settings.suppress_warnings:
        warnings.filterwarnings("ignore")

    #recording_paths = []
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort10_october2023/of") if f.is_dir()])
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort10_october2023/vr") if f.is_dir()])
    #parameter_helper_path = "/mnt/datastore/Harry/Cohort10_october2023/parameter_helper.csv"
    #process_recordings(recording_paths, parameter_helper_path=parameter_helper_path, allow_overwrite=True)

    #recording_paths = []
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort9_february2023/of") if f.is_dir()])
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort9_february2023/vr") if f.is_dir()])
    #parameter_helper_path = "/mnt/datastore/Harry/Cohort9_february2023/parameter_helper.csv"
    #process_recordings(recording_paths, parameter_helper_path=parameter_helper_path, allow_overwrite=True)

    #recording_paths = []
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])
    #parameter_helper_path = "/mnt/datastore/Harry/cohort8_may2021/parameter_helper.csv"
    #process_recordings(recording_paths, parameter_helper_path=parameter_helper_path, allow_overwrite=True)

    #recording_paths = []
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    #parameter_helper_path = "/mnt/datastore/Harry/cohort7_october2020/parameter_helper.csv"
    #process_recordings(recording_paths, parameter_helper_path=parameter_helper_path, allow_overwrite=True)

    #recording_paths = []
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])
    #recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    #parameter_helper_path = "/mnt/datastore/Harry/cohort6_july2020/parameter_helper.csv"
    #process_recordings(recording_paths, parameter_helper_path=parameter_helper_path, allow_overwrite=True)

    #recording_paths = []
    #parameter_helper_path = "/mnt/datastore/Harry/Cohort9_february2023/parameter_helper.csv"
    #process_recordings(recording_paths, parameter_helper_path=parameter_helper_path, allow_overwrite=True)

    recording_paths = []
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort11_april2024/vr") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort11_april2024/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort11_april2024/allen_brain_observatory_visual_coding") if f.is_dir()])
    parameter_helper_path = "/mnt/datastore/Harry/cohort11_april2024/parameter_helper.csv"
    process_recordings(recording_paths, parameter_helper_path=parameter_helper_path, allow_overwrite=True)



if __name__ == '__main__':
    main()