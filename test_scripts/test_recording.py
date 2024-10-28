import sys
import traceback
import warnings
import Elrond.settings as settings
import os
from Helpers.upload_download import get_recording_format, load_recording

def test_recording_with_spike_interface(recording_paths):
    """
    :param recording_paths: list of paths to recordings from which to process
    :return: None: if recording is successfully loaded with spike interface it will tell you :)
    """
    n_tested = len(recording_paths)
    n_passed_test = 0

    for recording_path in recording_paths:
        print("I will process recording ", recording_path)
        print("I will now try to load the recording as a recording extractor using spike interface")
        passed = False
        try:
            format = get_recording_format(recording_path)
            recording = load_recording(recording_path, format)
            print(recording)
            passed = True

        except Exception as ex:
            print('There was a problem! This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)

        if passed:
            n_passed_test += 1

    print("out of", str(n_tested), ", ", str(n_passed_test),
          "(",100*(n_passed_test/n_tested),"%) could be loaded successfully")
    print("I have tested all the recordings")
    return


def main():
    if settings.suppress_warnings:
        warnings.filterwarnings("ignore")

    # take a list of recordings to process
    # e.g. recording_paths = ["/mnt/datastore/Harry/test_recording/vr/M11_D36_2021-06-28_12-04-36"] or
    #      recording_paths = []
    #      recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/test_recording/vr") if f.is_dir()])
    # to grab a whole directory of recordings

    recording_paths = ['/mnt/datastore/Harry/cohort6_july2020/vr/M1_D6_2020-08-10_14-17-21',
                       '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D29_2021-06-17_11-50-37',
                       '/mnt/datastore/Harry/cohort8_may2021/of/M12_D29_2021-06-17_10-31-00',
                       '/mnt/datastore/Harry/cohort7_october2020/vr/M3_D6_2020-11-05_14-37-17',
                       '/mnt/datastore/Harry/cohort6_july2020/vr/M1_D9_2020-08-13_15-16-48',
                       '/mnt/datastore/Harry/cohort8_may2021/of/M14_D13_2021-05-26_10-51-36']
    test_recording_with_spike_interface(recording_paths)

    recording_paths = []
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])

    recording_paths = []
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort9_february2023/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort9_february2023/vr") if f.is_dir()])
    test_recording_with_spike_interface(recording_paths)

    recording_paths = []
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort10_october2023/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/Cohort10_october2023/vr") if f.is_dir()])
    test_recording_with_spike_interface(recording_paths)


if __name__ == '__main__':
    main()