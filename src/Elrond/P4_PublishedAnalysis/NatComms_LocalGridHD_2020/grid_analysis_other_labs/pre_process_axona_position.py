import glob
import numpy as np
import os
import PostSorting.open_field_spatial_data
import pandas as pd

server_path = '//cmvm.datastore.ed.ac.uk/cmvm/sbms/groups/mnolan_NolanLab/ActiveProjects/Tizzy/Cohorts1-2/'


def convert_axona_sync_pulses_to_continuous(axona_data):
    sync_pulse_times = axona_data.inp_data.times[1:]
    length = len(axona_data.tracking.times)
    sync_data = np.zeros(length)
    pulse_indices = (axona_data._inp_data.times[1:] * 50).astype(int)
    sync_data[pulse_indices] = 1
    return sync_data


def read_position_axona(path_to_position_file):
    import pyxona
    position_data = pd.DataFrame()
    axona_data = pyxona.File(path_to_position_file)
    position_data['time'] = axona_data.tracking.times
    position_data.time = position_data.time - position_data.time[1] + (position_data.time[2] - position_data.time[1])
    position_data.time[0] = 0
    position_data['time_seconds'] = position_data.time
    position_data['date'] = str(axona_data._start_datetime).split(' ')[0]
    position_data['x_left'] = axona_data.tracking.positions[:, 0]
    position_data['y_left'] = axona_data.tracking.positions[:, 1]
    position_data['x_right'] = axona_data.tracking.positions[:, 2]
    position_data['y_right'] = axona_data.tracking.positions[:, 3]
    sync_data = convert_axona_sync_pulses_to_continuous(axona_data)
    position_data['syncLED'] = sync_data
    # find and add sync data! axona_data.inp_data  # this just contains a few time stamps-convert based on matlab script
    return position_data


def find_axona_position_file(recording_folder):
    import pyxona
    if os.path.isdir(recording_folder) is False:
        print('    Error in open_field_spatial_data.py : The folder you specified as recording folder does not exist, please check if the path is correct.')
    path_to_axona_file = ''
    is_found = False
    for name in glob.glob(recording_folder + '/*.set'):
        if os.path.exists(name):
            try:
                path_to_axona_file = name
                pyxona.File(path_to_axona_file)
                is_found = True
                return path_to_axona_file, is_found
            except Exception as ex:
                print('Could not read axona file:')
                print(ex)

    return path_to_axona_file, is_found


def process_axona_recordings():
    axona_folder = False
    for dir, sub_dirs, files in os.walk(server_path):
        os.path.isdir(dir)
        for subdir in sub_dirs:
            for file in glob.glob(dir + '/' + subdir + '*'):
                axona_files = glob.glob(os.path.join(file, '*.set'))
                if len(axona_files) > 0:
                    axona_folder = True

                    if axona_folder:
                        try:
                            path_to_position_file, is_found = find_axona_position_file(file + '/')
                            position_data = read_position_axona(path_to_position_file)
                            position_data.to_pickle(file + '/axona_position.pkl')
                            print('saved ' + file + '/axona_position.pkl')
                        except:
                            print('did not manage to process ' + file)


def main():
    process_axona_recordings()


if __name__ == '__main__':
    main()

