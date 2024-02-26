from __future__ import division
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from Helpers import OpenEphys, open_ephys_IO
import settings

def load_sync_data_ephys(recording_to_process, prm):
    is_found = False
    sync_data = None
    print('loading sync channel...')
    file_path = recording_to_process + '/' + prm.get_sync_channel()
    if os.path.exists(file_path):
        sync_data = open_ephys_IO.get_data_continuous(file_path)
        is_found = True
    else:
        print('Sync data was not found, I will check if Axona sync data is present and convert it if it is.')
        events_file = recording_to_process + '/all_channels.events'
        if os.path.exists(events_file):
            events = OpenEphys.load(events_file)
            time_stamps = events['timestamps']
            channel = events['channel']
            pulse_indices = time_stamps[np.where(channel == 0)]
            # for sample in pulse_indices:   # make pulse wider
                # pulse_indices = np.append(pulse_indices, np.arange(sample, (sample + 5000)))

            # load any continuous data file to get length of recording
            for name in glob.glob(recording_to_process + '/*.continuous'):
                if os.path.exists(name):
                    print(name)
                    ch = open_ephys_IO.get_data_continuous(name)
                    length = len(ch)
                    sync_data = np.zeros(length)
                    sync_data[np.take(pulse_indices, np.where(pulse_indices < len(ch))).astype(int)] = 1
                    is_found = True
                    return sync_data, is_found

    return sync_data, is_found


def get_video_sync_on_and_off_times(spatial_data):
    threshold = np.median(spatial_data['syncLED']) + 4 * np.std(spatial_data['syncLED'])
    spatial_data['sync_pulse_on'] = spatial_data['syncLED'] > threshold
    spatial_data['sync_pulse_on_diff'] = np.append([None], np.diff(spatial_data['sync_pulse_on'].values))
    return spatial_data


def get_ephys_sync_on_and_off_times(sync_data_ephys):
    sync_data_ephys['on_index'] = sync_data_ephys['sync_pulse'] > 0.5
    sync_data_ephys['on_index_diff'] = np.append([None], np.diff(sync_data_ephys['on_index'].values))  # true when light turns on
    sync_data_ephys['time'] = sync_data_ephys.index / settings.sampling_rate
    return sync_data_ephys


def reduce_noise(pulses, threshold, high_level=5):
    '''
    Clean up the signal by assigning value lower than the threshold to 0
    and those higher than the threshold the high_level. The high_level is set to 5 by default
    to match with the oe signal. Setting the high_level is necessary because signal drift in the bonsai high level
    may lead to uneven weighting of the value in the correlation calculation
    '''

    pulses[pulses < threshold] = 0
    pulses[pulses >= threshold] = 5
    return pulses


def pad_shorter_array_with_0s(array1, array2):
    if len(array1) < len(array2):
        array1 = np.pad(array1, (0, len(array2)-len(array1)), 'constant')
    if len(array2) < len(array1):
        array2 = np.pad(array2, (0, len(array1)-len(array2)), 'constant')
    return array1, array2


def downsample_ephys_data(sync_data_ephys, spatial_data):
    avg_sampling_rate_bonsai = float(1 / spatial_data['time_seconds'].diff().mean())
    avg_sampling_rate_open_ephys = float(1 / sync_data_ephys['time'].diff().mean())
    sampling_rate_rate = avg_sampling_rate_open_ephys/avg_sampling_rate_bonsai
    length = int(len(sync_data_ephys['time']) / sampling_rate_rate)
    indices = (np.arange(length) * sampling_rate_rate).astype(int)
    sync_data_ephys_downsampled = sync_data_ephys['time'][indices]
    sync_data_ephys_downsampled['sync_pulse'] = sync_data_ephys['sync_pulse'][indices]
    sync_data_ephys_downsampled['time'] = sync_data_ephys['time'][indices]
    return sync_data_ephys_downsampled


def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index


# this is to remove any extra pulses that one dataset has but not the other
def trim_arrays_find_starts(sync_data_ephys_downsampled, spatial_data):
    oe_time = sync_data_ephys_downsampled.time
    bonsai_time = spatial_data.synced_time_estimate
    ephys_start_index = 19*30  # bonsai sampling rate times 19 seconds
    ephys_start_time = oe_time.values[19*30]
    bonsai_start_index = find_nearest(bonsai_time.values, ephys_start_time)
    return ephys_start_index, bonsai_start_index


#  this is needed for finding the rising edge of the pulse to by synced
def detect_last_zero(signal):
    '''
    signal is a already thresholded binary signal with 0 and 1
    return the index of the last 0 before the first 1
    '''
    first_index_in_signal = np.argmin(signal) # index of first zero value
    first_zero_index_in_signal = np.nonzero(signal)[0][0] #index of first non-zero value
    first_nonzero_index = first_index_in_signal + first_zero_index_in_signal # potential bug here if first_index_in_signal is not 0
    assert first_nonzero_index == first_zero_index_in_signal, 'Error, sync signal does not start at zero'
    last_zero_index = first_nonzero_index - 1
    return last_zero_index


def save_plots_of_pulses(bonsai, oe, output_path, lag, name='sync_pulses'):
    save_path = output_path + '/Figures/Sync_test/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.figure()
    bonsai_norm = bonsai / np.linalg.norm(bonsai)
    plt.plot(oe, color='red', label='open ephys')
    plt.plot(bonsai_norm * 3.5, color='black', label='bonsai')
    plt.title('lag=' + str(lag))
    plt.legend()
    plt.savefig(save_path + name + '_sync_pulses.png')
    plt.close()


def save_plot(prm, data, name, plot_color='black'):
    save_path = prm.get_output_path() + '/Figures/Sync_test/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.figure()
    plt.plot(data, color=plot_color, label=name)
    plt.legend()
    plt.savefig(save_path + name + '_sync_pulses.png')
    plt.close()


def get_synchronized_spatial_data(sync_data_ephys, spatial_data):
    """
    The ephys and spatial data is synchronized based on sync pulses sent both to the open ephys and bonsai systems.
    The open ephys GUI receives TTL pulses. Bonsai detects intensity from an LED that lights up whenever the TTL is
    sent to open ephys. The pulses have 20-60 s long randomised gaps in between them. The recordings don't necessarily
    start at the same time, so it is possible that bonsai will have an extra pulse that open ephys does not.
    Open ephys samples at 30000 Hz, and bonsai at 30 Hz, but the webcam frame rate is not precise.

    (1) I downsampled the open ephys signal to match the sampling rate of bonsai calculated based on the average
    interval between time stamps.
    (2) I reduced the noise in both signals by setting a threshold and replacing low values with 0s.
    (3) I calculated the correlation between the OE and Bonsai pulses
    (4) I calculated a lag estimate between the two signals based on the highest correlation.
    (5) I shifted the bonsai times by the lag.
    This reduces the delay to <100ms, so the pulses are more or less aligned at this point. The precision is lost because
    of the way I did the downsampling and the variable frame rate of the camera.
    (6) I cut the first 20 seconds of both arrays to make sure the first pulse of the array has a corresponding pulse
    from the other dataset.
    (7) I detected the rising edges of both peaks and subtracted the corresponding time values to get the lag.
    (8) I shifted the bonsai data again by this lag.
    Now the lag is within/around 30ms, so around the frame rate of the camera.
    Eventually, the shifted 'synced' times are added to the spatial dataframe.

    #Note: the syncLED column must have stable sampling frequency/FPS, otherwise there will be error
    """

    print('I will synchronize the position and ephys data by shifting the position to match the ephys.')
    sync_data_ephys_downsampled = downsample_ephys_data(sync_data_ephys, spatial_data)
    bonsai = spatial_data['syncLED'].values
    bonsai = np.append(0, np.diff(spatial_data['syncLED'].values)) # step to remove human error-caused light intensity jumps
    oe = sync_data_ephys_downsampled.sync_pulse.values
    bonsai = reduce_noise(bonsai, np.median(bonsai) + 6 * np.std(bonsai))
    oe = reduce_noise(oe, 2)
    bonsai, oe = pad_shorter_array_with_0s(bonsai, oe)
    corr = np.correlate(bonsai, oe, "full")  # this is the correlation array between the sync pulse series
    avg_sampling_rate_bonsai = float(1 / spatial_data['time_seconds'].diff().mean())
    lag = (np.argmax(corr) - (corr.size + 1)/2)/avg_sampling_rate_bonsai  # lag between sync pulses is based on max correlation
    spatial_data['synced_time_estimate'] = spatial_data.time_seconds - lag  # at this point the lag is about 100 ms

    # cut off first 19 seconds to make sure there will be a corresponding pulse
    ephys_start, bonsai_start = trim_arrays_find_starts(sync_data_ephys_downsampled, spatial_data)
    trimmed_ephys_time = sync_data_ephys_downsampled.time.values[ephys_start:]
    trimmed_ephys_pulses = oe[ephys_start:len(trimmed_ephys_time)]
    trimmed_bonsai_time = spatial_data['synced_time_estimate'].values[bonsai_start:]
    trimmed_bonsai_pulses = bonsai[bonsai_start:]

    oe_rising_edge_index = detect_last_zero(trimmed_ephys_pulses)
    oe_rising_edge_time = trimmed_ephys_time[oe_rising_edge_index]

    bonsai_rising_edge_index = detect_last_zero(trimmed_bonsai_pulses)
    bonsai_rising_edge_time = trimmed_bonsai_time[bonsai_rising_edge_index]

    lag2 = oe_rising_edge_time - bonsai_rising_edge_time
    print(f'Rising edge lag is {lag2}')

    spatial_data['synced_time'] = spatial_data.synced_time_estimate + lag2
    return spatial_data


def remove_opto_tagging_from_spatial_data(start_of_opto_tagging, spatial_data):
    if start_of_opto_tagging is None:
        return spatial_data
    else:
        start_of_opto_seconds = int(start_of_opto_tagging / settings.sampling_rate)
        nearest_bonsai_index = (np.abs(spatial_data.synced_time - start_of_opto_seconds)).argmin()
        spatial_data.drop(range(nearest_bonsai_index, len(spatial_data)), inplace=True)
    return spatial_data


def process_sync_data(recording_to_process, spatial_data):


    sync_data, is_found = load_sync_data_ephys(recording_to_process)
    sync_data_ephys = pd.DataFrame(sync_data)
    sync_data_ephys.columns = ['sync_pulse']
    sync_data_ephys = get_ephys_sync_on_and_off_times(sync_data_ephys)
    spatial_data = get_video_sync_on_and_off_times(spatial_data)
    spatial_data = get_synchronized_spatial_data(sync_data_ephys, spatial_data)
    # synced time in seconds, x and y in cm, hd in degrees
    synced_spatial_data = spatial_data[['synced_time', 'position_x', 'position_x_pixels',
                                        'position_y', 'position_y_pixels', 'hd', 'speed']].copy()
    # remove negative time points
    synced_spatial_data = synced_spatial_data.drop(synced_spatial_data[synced_spatial_data.synced_time < 0].index)
    synced_spatial_data = synced_spatial_data.reset_index(drop=True)
    return synced_spatial_data

def search_for_file(folder_to_search_in, string_to_find):
    matches = []
    for file in glob.glob(folder_to_search_in):
        if string_to_find in file:
            matches.append(file)
    if len(matches) == 0:
        return matches[0]
    else:
        print("I found multiple files that match the file being searched for")
        return matches


def get_ttl_pulse_array(recording_path):
    # first look in the paramfile
    # TODO

    # if still not found, use what is in the settings
    ttl_pulse_channel_path = search_for_file(recording_path, settings.ttl_pulse_channel)

    if ttl_pulse_channel_path.endswith(".continuous"):
        ttl_pulses = open_ephys_IO.get_data_continuous(ttl_pulse_channel_path)
        return ttl_pulses
    else:
        print("I don't know how to handle this ttl pulse file")
        return ""

def synchronise_position_data_via_ttl_pulses(position_data, recording_path):
    # get array for the ttl pulses in the ephys data
    sync_data = get_ttl_pulse_array(recording_path)
    sync_data_ephys = pd.DataFrame(sync_data)
    sync_data_ephys.columns = ['sync_pulse']
    sync_data_ephys = get_ephys_sync_on_and_off_times(sync_data_ephys)

    position_data = get_video_sync_on_and_off_times(position_data)
    position_data = get_synchronized_spatial_data(sync_data_ephys, position_data)
    # synced time in seconds, x and y in cm, hd in degrees
    synced_position_data = position_data[['synced_time', 'position_x', 'position_x_pixels',
                                        'position_y', 'position_y_pixels', 'hd', 'speed']].copy()
    # remove negative time points
    synced_position_data = synced_position_data.drop(synced_position_data[synced_position_data.synced_time < 0].index)
    synced_position_data = synced_position_data.reset_index(drop=True)

    return synced_position_data

