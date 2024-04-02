from Helpers.open_ephys_IO import OpenEphys
import numpy  as np

def load_OpenEphysRecording_fix_reader_and_write(folder, channel_ids=None):
    if channel_ids is None:
        channel_ids = np.arange(1, 16+1)

    all_first_timestamps = []
    all_time_stamps = []
    all_last_timestamps = []
    for i, channel_id in enumerate(channel_ids):
        fname = folder+"/100_CH"+str(channel_id)+'.continuous'
        data = OpenEphys.loadContinuousFast(fname)

        if i==0:
            og_timestamps = data["timestamps"]
        timestamps = data["timestamps"]
        all_time_stamps.append(timestamps)
        all_first_timestamps.append(timestamps[0])
        all_last_timestamps.append(timestamps[-1])

        OpenEphys.writeContinuousFile(fname, data['header'], og_timestamps, data['data'], data['recordingNumber'])

    for i in range(8):
        fname = folder + "/100_ADC" + str(i+1) + '.continuous'
        data = OpenEphys.loadContinuousFast(fname)
        all_time_stamps.append(timestamps)
        all_first_timestamps.append(timestamps[0])
        all_last_timestamps.append(timestamps[-1])
        OpenEphys.writeContinuousFile(fname, data['header'], og_timestamps, data['data'], data['recordingNumber'])

    print("")
    for i in range(16+8):
        for j in range(16+8):
            a = np.all(all_time_stamps[i] == all_time_stamps[j])
            if a == False:
                print("OH")

    if not all(all_first_timestamps[0] == e for e in all_first_timestamps) or not all(
        all_last_timestamps[0] == e for e in all_last_timestamps):
        print("")





def main():

    recording_paths = ["/mnt/datastore/Harry/test_recording/vr/M11_D36_2021-06-28_12-04-36"] # example vr tetrode session with a linked of session
    recording_paths = ["/mnt/datastore/Harry/test_recording/vr/M18_D1_2023-10-30_12-38-29"]
    recording_paths = ["/mnt/datastore/Harry/cohort6_july2020/vr/M1_D6_2020-08-10_14-17-21"]
    #recording_paths = ["/home/ubuntu/to_sort/recordings/M16_D1_2023-02-28_17-42-27"]
    #recording_paths=["/mnt/datastore/Harry/Cohort9_february2023/of/M16_D1_2023-02-28_18-42-28"]

    load_OpenEphysRecording_fix_reader_and_write(recording_paths[0])
    print("H!")

if __name__ == '__main__':
    main()