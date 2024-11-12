import traceback
import sys
from os import listdir
from os.path import isfile, join
from Elrond.Helpers.OpenEphys import *
from Elrond.Helpers.upload_download import get_recording_format


def consecutive_zeros_mask(arr, n):
    consecutive_zeros_mask = np.zeros_like(arr, dtype=bool)
    zero_indices = np.nonzero((arr == 0) * 1)[0]
    for zi in zero_indices:
        count = 1
        while (count < len(arr)-zi) and arr[zi+count] == 0.:
            count = count + 1
        if count >= n:
            consecutive_zeros_mask[zi:zi+count] = True
    return consecutive_zeros_mask


def repair_gaps_in_OpenEphysLegacy_recordings(recording_paths):
    for recording_path in recording_paths:
        try:
            print("I will process recording ", recording_path)
            format = get_recording_format(recording_path)
            if format == "openephys":
                continuous_files = [f for f in listdir(recording_path) if (isfile(join(recording_path, f)) & (f.endswith(".continuous")))]

                # find common zero gaps in channel files
                some_channel_continuous_files = list(filter(lambda k: '_CH' in k, continuous_files))[:3]
                data = []
                for cont_file in some_channel_continuous_files:
                    cont_path = recording_path + "/" + cont_file
                    data_dict = loadContinuousFast(cont_path, dtype=np.int16)
                    raw_data = data_dict['data']
                    data.append(raw_data)
                data = np.array(data)
                avg_data = np.nanmean(data, axis=0, dtype=np.int16)
                # get zero mask with some tolerance
                zero_mask = consecutive_zeros_mask(avg_data, n=10)

                for cont_file in continuous_files:
                    cont_path = recording_path+"/"+cont_file
                    data_dict = loadContinuous(cont_path, dtype=np.int16)
                    raw_data = data_dict['data']
                    timestamps = data_dict['timestamps']
                    new_raw_data = data_dict['data'][~zero_mask]
                    new_timestamps = (timestamps[0] + np.arange(0, len(new_raw_data) + 1024, 1024))

                    # rewrite without gap
                    f = open(cont_path, 'rb')
                    header = readHeader(f)
                    writeContinuousFile(cont_path, header, new_timestamps, new_raw_data, dtype=np.int16)

        except Exception as ex:
            print('There was a problem! This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)

def main():
    print("================================================================================")
    print("================================================================================")
    print("=====================                                         ==================")
    print("=====================       WARNING: THIS SCRIPT EDITS        ==================")
    print("=====================       RAW DATA! USE WITH CAUTION        ==================")
    print("=====================                                         ==================")
    print("================================================================================")
    print("================================================================================")

    #recording_paths = ['/mnt/datastore/Harry/cohort6_july2020/vr/M1_D6_2020-08-10_14-17-21',
    #                   '/mnt/datastore/Harry/cohort8_may2021/vr/M13_D29_2021-06-17_11-50-37',
    #                   '/mnt/datastore/Harry/cohort8_may2021/of/M12_D29_2021-06-17_10-31-00',
    #                   '/mnt/datastore/Harry/cohort7_october2020/vr/M3_D6_2020-11-05_14-37-17',
    #                   '/mnt/datastore/Harry/cohort6_july2020/vr/M1_D9_2020-08-13_15-16-48',
    #                   '/mnt/datastore/Harry/cohort8_may2021/of/M14_D13_2021-05-26_10-51-36']
    #repair_gaps_in_OpenEphysLegacy_recordings(recording_paths)
    print("")


if __name__ == '__main__':
    main()