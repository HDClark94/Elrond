import sys
import traceback
import warnings
import settings
import os
import spikeinterface.full as si


def test_recordings_with_spike_interface(recording_paths, nwb_folder_name= "", suffix=""):
    """
    :param recording_paths: list of paths to recordings from which to process
    :param nwb_folder_name: name of the folder all the nwb results will be returned to
    :return: None: if recording is successfully loaded with spike interface it will tell you :)
    """
    n_tested = len(recording_paths)
    n_passed_test = 0

    for recording_path in recording_paths:
        recording_name = os.path.basename(recording_path)
        print("I will process recording ", recording_path)
        print("I will now try to load the nwb file as a recording extractor using spike interface")
        nwb_path = recording_path + "/" + nwb_folder_name
        nwb_file = nwb_path + "/" + recording_name + suffix + ".nwb"
        passed = False
        try:
            recording = si.read_nwb_recording(nwb_file)
            print(recording)
            print(recording.get_channel_locations())
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

    recording_paths = []
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])

    test_recordings_with_spike_interface(recording_paths, nwb_folder_name="processed/nwb/", suffix="_v3")

if __name__ == '__main__':
    main()