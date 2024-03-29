import os
import glob
import numpy as np
import pandas as pd
from Helpers import open_ephys_IO
from Helpers.array_utility import *
import settings


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
    to match with the ephys signal. Setting the high_level is necessary because signal drift in the bonsai high level
    may lead to uneven weighting of the value in the correlation calculation
    '''

    pulses[pulses < threshold] = 0
    pulses[pulses >= threshold] = high_level
    return pulses


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


# this is to remove any extra pulses that one dataset has but not the other
def trim_arrays_find_starts(sync_data_ephys_downsampled, spatial_data):
    ephys_time = sync_data_ephys_downsampled.time
    bonsai_time = spatial_data.synced_time_estimate
    ephys_start_index = 19*30  # bonsai sampling rate times 19 seconds
    ephys_start_time = ephys_time.values[19*30]
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


def calculate_lag(sync_data_ephys, spatial_data):
    """
    The ephys and spatial data is synchronized based on sync pulses sent both to the open ephys and bonsai systems.
    The open ephys GUI receives TTL pulses. Bonsai detects intensity from an LED that lights up whenever the TTL is
    sent to open ephys. The pulses have 20-60 s long randomised gaps in between them. The recordings don't necessarily
    start at the same time, so it is possible that bonsai will have an extra pulse that open ephys does not.
    Open ephys samples at 30000 Hz, and bonsai at 30 Hz, but the webcam frame rate is not precise.

    (1) I downsampled the open ephys signal to match the sampling rate of bonsai calculated based on the average
    interval between time stamps.
    (2) I reduced the noise in both signals by setting a threshold and replacing low values with 0s.
    (3) I calculated the correlation between the ephys and Bonsai pulses
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
    bonsai = np.append(0, np.diff(spatial_data['syncLED'].values)) # step to remove human error-caused light intensity jumps
    ephys = sync_data_ephys_downsampled.sync_pulse.values
    bonsai = reduce_noise(bonsai, np.median(bonsai) + 6 * np.std(bonsai))
    ephys = reduce_noise(ephys, 2)
    bonsai, ephys = pad_shorter_array_with_0s(bonsai, ephys)
    corr = np.correlate(bonsai, ephys, "full")  # this is the correlation array between the sync pulse series
    avg_sampling_rate_bonsai = float(1 / spatial_data['time_seconds'].diff().mean())
    lag = (np.argmax(corr) - (corr.size + 1)/2)/avg_sampling_rate_bonsai  # lag between sync pulses is based on max correlation
    spatial_data['synced_time_estimate'] = spatial_data.time_seconds - lag  # at this point the lag is about 100 ms

    # cut off first 19 seconds to make sure there will be a corresponding pulse
    ephys_start, bonsai_start = trim_arrays_find_starts(sync_data_ephys_downsampled, spatial_data)
    trimmed_ephys_time = sync_data_ephys_downsampled.time.values[ephys_start:]
    trimmed_ephys_pulses = ephys[ephys_start:len(trimmed_ephys_time)]
    trimmed_bonsai_time = spatial_data['synced_time_estimate'].values[bonsai_start:]
    trimmed_bonsai_pulses = bonsai[bonsai_start:]

    ephys_rising_edge_index = detect_last_zero(trimmed_ephys_pulses)
    ephys_rising_edge_time = trimmed_ephys_time[ephys_rising_edge_index]

    bonsai_rising_edge_index = detect_last_zero(trimmed_bonsai_pulses)
    bonsai_rising_edge_time = trimmed_bonsai_time[bonsai_rising_edge_index]

    lag2 = ephys_rising_edge_time - bonsai_rising_edge_time
    print(f'Rising edge lag is {lag2}')
    return lag2


def search_for_file(folder_to_search_in, string_to_find):
    matches = []
    for file_path in glob.glob(folder_to_search_in + "/**", recursive=True):
        if string_to_find in file_path:
            matches.append(file_path)
    return matches

def get_ttl_pulse_array(recording_path):
    # first look in the paramfile
    # TODO

    # if still not found, use what is in the settings
    ttl_pulse_channel_paths = search_for_file(recording_path, settings.ttl_pulse_channel)
    assert len(ttl_pulse_channel_paths) == 1
    ttl_pulse_channel_path = ttl_pulse_channel_paths[0]

    if ttl_pulse_channel_path.endswith(".continuous"):
        ttl_pulses = open_ephys_IO.get_data_continuous(ttl_pulse_channel_path)
        ttl_pulses = pd.DataFrame(ttl_pulses)
        ttl_pulses.columns = ['sync_pulse']
        return ttl_pulses
    else:
        print("I don't know how to handle this ttl pulse file")
        return ""

def synchronise_position_data_via_ttl_pulses(position_data, recording_path):
    # get array for the ttl pulses in the ephys data
    sync_data = get_ttl_pulse_array(recording_path)

    # get on and off times for ttl pulse
    sync_data = get_ephys_sync_on_and_off_times(sync_data)
    position_data = get_video_sync_on_and_off_times(position_data)

    # calculate lag and align the position data
    lag = calculate_lag(sync_data, position_data)
    position_data['synced_time'] = position_data.synced_time_estimate + lag

    # remove negative time points
    position_data = position_data.drop(position_data[position_data.synced_time < 0].index)
    position_data = position_data.reset_index(drop=True)

    return position_data

