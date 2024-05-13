import os
import glob
import math
import settings
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from scipy.interpolate import interp1d
from Helpers import math_utility

def read_bonsai_file(recording_folder):
    if os.path.isdir(recording_folder) is False:
        print('I could not find a bonsai file in', recording_folder)
    path_to_bonsai_file = ''
    for name in glob.glob(recording_folder + '/*.csv'):
        if os.path.exists(name):
            with open(name, newline='') as file:
                try:
                    reader = csv.reader(file)
                    row1 = next(reader)
                    if "T" not in row1[0]:
                        continue
                    else:
                        if len(row1[0].split('T')[0]) == 10:
                            path_to_bonsai_file = name
                except Exception as ex:
                    print('Could not read csv file:')
                    print(name)
                    print(ex)

    return pd.read_csv(path_to_bonsai_file, sep=' ', header=None)



''' 
Read raw position data and sync LED intensity from Bonsai file amd convert time to seconds
'''

def convert_time_to_seconds(position_data):
    position_data[['hours','minutes','seconds']] = position_data['time'].str.split(':', n=2,expand=True)
    position_data['hours'] = position_data['hours'].astype(int)
    position_data['minutes'] = position_data['minutes'].astype(int)
    position_data['seconds'] = position_data['seconds'].astype(float)
    position_data['time_seconds'] = position_data['hours'] * 3600 + position_data['minutes']*60 + position_data['seconds']
    position_data['time_seconds'] = position_data['time_seconds'] - position_data['time_seconds'][0]
    return position_data


def resample_position_data(position_data, fs):
    '''
    Resample position data so FPS is consistent.
    Sometimes the FPS of the camera is not stable,
    which may lead to error in syncing.
    Assume pos has a time_seconds column
    '''

    t = position_data.time_seconds.values
    t2 = np.arange(0,t[-1],1/fs)

    print('I will now resample the data at ',str(fs), ' Hz')
    print("Mean frame rate:", str(len(position_data)/(t[-1]-t[0])), "FPS")
    print("SD of frame rate:", str(np.nanstd(1 / np.diff(t))), "FPS")

    df = {}
    for col in position_data.columns:
        f = interp1d(t, position_data[col].values)
        df[col] = f(t2)
    df['time_seconds'] = t2
    df2return = pd.DataFrame(df)
    return df2return


def proces_bonsai_position(position_data):
    if len(list(position_data)) > 6:
        position_data = position_data.drop([6], axis=1)  # remove column of NaNs due to extra space at end of lines
    position_data[['date','time']] = position_data[0].str.split('T', n=1,expand=True)
    position_data[['time','str_to_remove']] = position_data['time'].str.split('+', n=1,expand=True)
    position_data = position_data.drop([0, 'str_to_remove'], axis=1)  # remove first column that got split into date and time
    position_data.columns = ['x_left', 'y_left', 'x_right', 'y_right', 'syncLED', 'date', 'time']
    position_data = convert_time_to_seconds(position_data)
    position_data = position_data.drop(['date', 'time', 'hours', 'minutes', 'seconds'], axis=1)
    return position_data


def calculate_speed(position_data):
    elapsed_time = position_data['time_seconds'].diff()
    distance_travelled_left = np.sqrt(position_data['x_left'].diff().pow(2) + position_data['y_left'].diff().pow(2))
    distance_travelled_right = np.sqrt(position_data['x_right'].diff().pow(2) + position_data['y_right'].diff().pow(2))

    position_data['speed_left'] = distance_travelled_left / elapsed_time
    position_data['speed_right'] = distance_travelled_right / elapsed_time
    return position_data


def calculate_central_speed(position_data):
    elapsed_time = position_data['time_seconds'].diff()
    distance_travelled = np.sqrt(position_data['position_x'].diff().pow(2) + position_data['position_y'].diff().pow(2))
    position_data['speed'] = distance_travelled / elapsed_time
    return position_data


def remove_jumps(position_data):
    max_speed = 1  # m/s, anything above this is not realistic
    pixel_ratio = settings.pixel_ratio
    max_speed_pixels = max_speed * pixel_ratio
    speed_exceeded_left = position_data['speed_left'] > max_speed_pixels
    position_data['x_left_without_jumps'] = position_data.x_left[speed_exceeded_left == False]
    position_data['y_left_without_jumps'] = position_data.y_left[speed_exceeded_left == False]

    speed_exceeded_right = position_data['speed_right'] > max_speed_pixels
    position_data['x_right_without_jumps'] = position_data.x_right[speed_exceeded_right == False]
    position_data['y_right_without_jumps'] = position_data.y_right[speed_exceeded_right == False]

    remains_left = (len(position_data) - speed_exceeded_left.sum())/len(position_data)*100
    remains_right = (len(position_data) - speed_exceeded_right.sum())/len(position_data)*100
    print('{} % of right side tracking data, and {} % of left side'
          ' remains after removing the ones exceeding speed limit.'.format(remains_right, remains_left))
    return position_data


def get_distance_of_beads(position_data):
    distance_between_beads = np.sqrt((position_data['x_left'] - position_data['x_right']).pow(2) + (position_data['y_left'] - position_data['y_right']).pow(2))
    return distance_between_beads


def remove_points_where_beads_are_far_apart(position_data):
    minimum_distance = 40
    distance_between_beads = get_distance_of_beads(position_data)
    distance_exceeded = distance_between_beads > minimum_distance
    position_data['x_left_cleaned'] = position_data.x_left_without_jumps[distance_exceeded == False]
    position_data['x_right_cleaned'] = position_data.x_right_without_jumps[distance_exceeded == False]
    position_data['y_left_cleaned'] = position_data.y_left_without_jumps[distance_exceeded == False]
    position_data['y_right_cleaned'] = position_data.y_right_without_jumps[distance_exceeded == False]
    return position_data


def curate_position(position_data):
    position_data = remove_jumps(position_data)
    position_data = remove_points_where_beads_are_far_apart(position_data)
    return position_data


def calculate_position(position_data):
    position_data['position_x_tmp'] = (position_data['x_left_cleaned'] + position_data['x_right_cleaned']) / 2
    position_data['position_y_tmp'] = (position_data['y_left_cleaned'] + position_data['y_right_cleaned']) / 2

    position_data['position_x'] = position_data['position_x_tmp'].interpolate()  # interpolate missing data
    position_data['position_y'] = position_data['position_y_tmp'].interpolate()
    return position_data


def calculate_head_direction(position_data):
    position_data['head_dir_tmp'] = np.degrees(np.arctan((position_data['y_left_cleaned'] + position_data['y_right_cleaned']) /
                                                         (position_data['x_left_cleaned'] + position_data['x_right_cleaned'])))
    rho, hd = math_utility.cart2pol(position_data['x_right_cleaned'] - position_data['x_left_cleaned'],
                                    position_data['y_right_cleaned'] - position_data['y_left_cleaned'])
    position_data['hd'] = np.degrees(hd)
    position_data['hd'] = position_data['hd'].interpolate()  # interpolate missing data
    return position_data


def convert_to_cm(position_data):
    pixel_ratio = settings.pixel_ratio
    position_data['position_x_pixels'] = position_data.position_x
    position_data['position_y_pixels'] = position_data.position_y
    position_data['position_x'] = position_data.position_x / pixel_ratio * 100
    position_data['position_y'] = position_data.position_y / pixel_ratio * 100
    return position_data


def shift_to_start_from_zero_at_bottom_left(position_data):
    position_data['position_x'] = position_data.position_x - min(position_data.position_x[~np.isnan(position_data.position_x)])
    position_data['position_y'] = position_data.position_y - min(position_data.position_y[~np.isnan(position_data.position_y)])
    return position_data


def get_dwell(spatial_data):
    min_dwell_distance_cm = 5  # from point to determine min dwell time
    min_dwell_distance_pixels = min_dwell_distance_cm/100 * settings.pixel_ratio
    dt_position_ms = spatial_data.synced_time.diff().mean() * 1000  # average sampling interval in position data (ms)
    min_dwell_time_ms = 3 * dt_position_ms  # this is about 100 ms
    min_dwell = round(min_dwell_time_ms / dt_position_ms)
    return min_dwell, min_dwell_distance_pixels


def get_bin_size():
    bin_size_cm = settings.open_field_bin_size_cm
    bin_size_pixels = bin_size_cm/100*settings.pixel_ratio
    return bin_size_pixels


def get_number_of_bins(spatial_data):
    bin_size_pixels = get_bin_size()
    length_of_arena_x = spatial_data.position_x_pixels[~np.isnan(spatial_data.position_x_pixels)].max()
    length_of_arena_y = spatial_data.position_y_pixels[~np.isnan(spatial_data.position_y_pixels)].max()
    number_of_bins_x = math.ceil(length_of_arena_x / bin_size_pixels)
    number_of_bins_y = math.ceil(length_of_arena_y / bin_size_pixels)
    return number_of_bins_x, number_of_bins_y


def get_position_heatmap(spatial_data):
    min_dwell, min_dwell_distance_pixels = get_dwell(spatial_data)
    bin_size_pixels = get_bin_size()
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data)

    position_heat_map = np.zeros((number_of_bins_x, number_of_bins_y))
    # find value for each bin for heatmap
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_pixels + (bin_size_pixels / 2)
            py = y * bin_size_pixels + (bin_size_pixels / 2)

            occupancy_distances = np.sqrt(np.power((px - spatial_data.position_x_pixels.values), 2)
                                        + np.power((py - spatial_data.position_y_pixels.values), 2))
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_pixels)[0])

            if bin_occupancy >= min_dwell:
                position_heat_map[x, y] = bin_occupancy
            else:
                position_heat_map[x, y] = np.nan
    return position_heat_map


def extract_position_from_dlc(recording_path):
    dlc_position_data = pd.DataFrame()
    files = [f for f in Path(recording_path).iterdir()]
    if np.any([".avi" in f.name and f.is_file() for f in files]):
        return dlc_position_data
    return dlc_position_data


def process_position_data(recording_path, **kwargs):
    bonsai_position_data = read_bonsai_file(recording_path)
    position_data = proces_bonsai_position(bonsai_position_data)

    if "use_dlc_to_extract_openfield_position" in kwargs:
        if kwargs["use_dlc_to_extract_openfield_position"]:
            dlc_position_data = extract_position_from_dlc(recording_path)
            position_data = pd.merge(position_data, dlc_position_data)

    position_data = resample_position_data(position_data, 30)

    position_data = calculate_speed(position_data)
    position_data = curate_position(position_data)  # remove jumps from data, and when the beads are far apart
    position_data = calculate_position(position_data)  # get central position and interpolate missing data
    position_data = calculate_head_direction(position_data)  # use coord from the two beads to get hd and interpolate
    position_data = shift_to_start_from_zero_at_bottom_left(position_data)
    position_data = convert_to_cm(position_data)
    position_data = calculate_central_speed(position_data)

    position_data = position_data[['time_seconds', 'position_x', 'position_x_pixels',
                                   'position_y', 'position_y_pixels', 'hd', 'syncLED', 'speed']]
    return position_data
