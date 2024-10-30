import os
import glob
import spikeinterface.full as si
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Elrond.Helpers import open_ephys_IO
from Elrond.Helpers.array_utility import *
import Elrond.settings as settings
from neuroconv.utils.dict import load_dict_from_file, dict_deep_update



def get_video_sync_on_and_off_times(video_data):
    threshold = np.median(video_data['syncLED']) + 4 * np.std(video_data['syncLED'])
    video_data['sync_pulse_on'] = video_data['syncLED'] > threshold
    video_data['sync_pulse_on_diff'] = np.append([None], np.diff(video_data['sync_pulse_on'].values))
    return video_data

def get_ephys_sync_on_and_off_times(sync_data_ephys):
    sync_data_ephys['on_index'] = sync_data_ephys['sync_pulse'] > 0.5
    sync_data_ephys['on_index_diff'] = np.append([None], np.diff(sync_data_ephys['on_index'].values))  # true when light turns on
    sync_data_ephys['time'] = sync_data_ephys.index / settings.sampling_rate
    return sync_data_ephys

def get_behaviour_sync_on_and_off_times(position_data):
    position_data['on_index'] = position_data['sync_pulse'] > 0.5
    position_data['on_index_diff'] = np.append([None], np.diff(position_data['on_index'].values))  # true when light turns on
    position_data['time'] = position_data["time_seconds"]
    return position_data

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
    sync_data_ephys_downsampled = pd.DataFrame()
    sync_data_ephys_downsampled['sync_pulse'] = sync_data_ephys['sync_pulse'][indices]
    sync_data_ephys_downsampled['time'] = sync_data_ephys['time'][indices]
    return sync_data_ephys_downsampled


# this is to remove any extra pulses that one dataset has but not the other
def trim_arrays_find_starts(sync_data_ephys_downsampled, spatial_data, trim_time_seconds=60):
    ephys_time = sync_data_ephys_downsampled.time
    spatial_time = spatial_data.synced_time_estimate
    spatial_data_sampling_rate = float(1/spatial_data['time_seconds'].diff().mean())
    ephys_start_index = int(trim_time_seconds*spatial_data_sampling_rate)  # bonsai sampling rate times trim_time in seconds
    ephys_start_time = ephys_time.values[int(trim_time_seconds*spatial_data_sampling_rate)]
    spatial_start_index = find_nearest(spatial_time.values, ephys_start_time)
    return ephys_start_index, spatial_start_index


#  this is needed for finding the rising edge of the pulse to by synced
def detect_last_zero(signal): 
    '''
    signal is a already thresholded binary signal with 0 and 1
    return the index of the last 0 before the first 1
    '''
    first_index_in_signal = np.argmin(signal) # index of first zero value
    first_zero_index_in_signal = np.nonzero(signal)[0][0] #index of first non-zero value
    first_nonzero_index = first_index_in_signal + first_zero_index_in_signal # potential bug here if first_index_in_signal is not 0
    #assert first_nonzero_index == first_zero_index_in_signal, 'Error, sync signal does not start at zero'
    last_zero_index = first_nonzero_index - 1
    return last_zero_index


def adjust_for_lag(sync_data_ephys, spatial_data, recording_path, processed_path):
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
    # first check if theres is a manually add offset in cases where TTL synchronisation failed and was known
    matches = search_for_file(recording_path, "lag.npy")
    if len(matches)==1:
        print("I have found a lag.npy file and will use the offset specified here")
        lag = np.load(matches[0]) 
        spatial_data['synced_time_estimate'] = spatial_data.time_seconds - lag
        spatial_data['synced_time'] = spatial_data.synced_time_estimate
        return spatial_data

    output_path = processed_path
    print('I will synchronize the position and ephys data by shifting the position to match the ephys.')
    sync_data_ephys_downsampled = downsample_ephys_data(sync_data_ephys, spatial_data)
    bonsai = np.append(0, np.diff(spatial_data['syncLED'].values)) # step to remove human error-caused light intensity jumps
    ephys = sync_data_ephys_downsampled.sync_pulse.values
    save_plots_of_pulses(bonsai=bonsai, output_path=output_path, lag=np.nan, name='bonsai')
    save_plots_of_pulses(ephys=ephys,   output_path=output_path, lag=np.nan, name='ephys')

    bonsai = reduce_noise(bonsai, np.median(bonsai) + 6 * np.std(bonsai))
    if max(ephys)>1:
        ephys = reduce_noise(ephys, 2)
    save_plots_of_pulses(bonsai=bonsai, ephys=ephys, output_path=output_path, lag=np.nan, name='pulses_before_processing')

    bonsai, ephys = pad_shorter_array_with_0s(bonsai, ephys)
    corr = np.correlate(bonsai, ephys, "full")  # this is the correlation array between the sync pulse series
    avg_sampling_rate_bonsai = float(1 / spatial_data['time_seconds'].diff().mean())
    lag = (np.argmax(corr) - (corr.size + 1)/2)/avg_sampling_rate_bonsai  # lag between sync pulses is based on max correlation
    spatial_data['synced_time_estimate'] = spatial_data.time_seconds - lag  # at this point the lag is about 100 ms

    # cut off first n seconds to make sure there will be a corresponding pulse
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
    if np.abs(lag2)<1: # i.e. a sensible adjustment for the rising edge
        spatial_data['synced_time'] = spatial_data.synced_time_estimate + lag2 
    else: # if too big then just use correlative lag
        spatial_data['synced_time'] = spatial_data.synced_time_estimate

    print(f'Rising edge lag is {lag2}')
    save_plots_of_pulses(bonsai=trimmed_bonsai_pulses, ephys=trimmed_ephys_pulses,
                         output_path=output_path, lag=lag2, name='pulses_after_processing')
    return spatial_data


def adjust_for_lag_ephys(sync_data_ephys_downsampled, spatial_data, recording_path, processed_path):
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
    # first check if theres is a manually add offset in cases where TTL synchronisation failed and was known
    matches = search_for_file(recording_path, "lag.npy")
    if len(matches)==1:
        print("I have found a lag.npy file and will use the offset specified here")
        lag = np.load(matches[0])[0]
        spatial_data['synced_time_estimate'] = spatial_data.time_seconds - lag
        spatial_data['synced_time'] = spatial_data.synced_time_estimate
        return spatial_data

    output_path = processed_path
    print('I will synchronize the position and ephys data by shifting the position to match the ephys.')
    bonsai = np.append(0, np.diff(spatial_data['syncLED'].values)) # step to remove human error-caused light intensity jumps
    ephys = sync_data_ephys_downsampled.sync_pulse.values
    save_plots_of_pulses(bonsai=bonsai, output_path=output_path, lag=np.nan, name='bonsai')
    save_plots_of_pulses(ephys=ephys,   output_path=output_path, lag=np.nan, name='ephys')

    bonsai = reduce_noise(bonsai, np.median(bonsai) + 6 * np.std(bonsai))
    if max(ephys)>1:
        ephys = reduce_noise(ephys, 2)
    save_plots_of_pulses(bonsai=bonsai, ephys=ephys, output_path=output_path, lag=np.nan, name='pulses_before_processing')

    bonsai, ephys = pad_shorter_array_with_0s(bonsai, ephys)
    corr = np.correlate(bonsai, ephys, "full")  # this is the correlation array between the sync pulse series
    avg_sampling_rate_bonsai = float(1 / spatial_data['time_seconds'].diff().mean())
    lag = (np.argmax(corr) - (corr.size + 1)/2)/avg_sampling_rate_bonsai  # lag between sync pulses is based on max correlation
    spatial_data['synced_time_estimate'] = spatial_data.time_seconds - lag  # at this point the lag is about 100 ms

    # cut off first n seconds to make sure there will be a corresponding pulse
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
    if np.abs(lag2)<1: # i.e. a sensible adjustment for the rising edge
        spatial_data['synced_time'] = spatial_data.synced_time_estimate + lag2 
    else: # if too big then just use correlative lag
        spatial_data['synced_time'] = spatial_data.synced_time_estimate

    print(f'Rising edge lag is {lag2}')
    save_plots_of_pulses(bonsai=trimmed_bonsai_pulses, ephys=trimmed_ephys_pulses,
                         output_path=output_path, lag=lag2, name='pulses_after_processing')
    return spatial_data

def save_plots_of_pulses(bonsai=None, ephys=None, output_path=None, lag=np.nan, name=""):
    save_path = output_path + 'Figures/Sync_test/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.figure()
    if ephys is not None:
        plt.plot(ephys, color='red', label='open ephys')
    if bonsai is not None:
        bonsai_norm = bonsai / np.linalg.norm(bonsai)
        plt.plot(bonsai_norm * 3.5, color='black', label='bonsai')
    plt.title('lag=' + str(lag))
    plt.legend()
    plt.savefig(save_path + name + '_sync_pulses.png')
    plt.close()

def search_for_file(folder_to_search_in, string_to_find):
    matches = []
    for file_path in glob.glob(folder_to_search_in + "/**", recursive=True):
        if string_to_find in file_path:
            matches.append(file_path)
    return matches

def get_ttl_pulse_array_in_column(df):
    if "syncLED" in df.columns:
        ttl_pulses = pd.DataFrame(df["syncLED"])
    elif "sync_pulse" in df.columns:
        ttl_pulses = pd.DataFrame(df["sync_pulse"])
    ttl_pulses.columns = ['sync_pulse']
    return ttl_pulses

def get_ttl_pulse_array_in_ADC_channel(recording_path):
    # retrieve TTL sync pulses from recording

    if os.path.exists(recording_path + "/params.yml"):
        params = load_dict_from_file(recording_path + "/params.yml")
        if ("probe_manufacturer" in params.keys()) and ("recording_aquisition" in params.keys()):
            if (params["probe_manufacturer"] == 'neuropixel') and (params["recording_aquisition"] == 'openephys'):
                recording = si.read_openephys(recording_path, load_sync_channel=True)
                ttl_pulses = recording.get_traces(channel_ids=[recording.get_channel_ids()[-1]])
                ttl_pulses = np.asarray(ttl_pulses)[:,0]
                ttl_pulses = pd.DataFrame(ttl_pulses)
                ttl_pulses.columns = ['sync_pulse']
                return ttl_pulses

    # if still not found, use what is in the settings
    ttl_pulse_channel_paths = search_for_file(recording_path, settings.ttl_pulse_channel)
    assert len(ttl_pulse_channel_paths) == 1
    ttl_pulse_channel_path = ttl_pulse_channel_paths[0]
    ttl_pulses = open_ephys_IO.get_data_continuous(ttl_pulse_channel_path)
    ttl_pulses = pd.DataFrame(ttl_pulses)
    ttl_pulses.columns = ['sync_pulse']
    return ttl_pulses

# Note: we'll need to change this when we switch to full zarr
def get_downsampled_ttl_pulse_array(recording_path, spatial_data, ephys_sampling_freq=30000):

    recording = si.read_openephys(recording_path, load_sync_channel=True)
    raw_sync_data = recording.get_traces(channel_ids=[recording.get_channel_ids()[-1]]) 

    avg_sampling_rate_bonsai = float(1 / spatial_data['time_seconds'].diff().mean())
    avg_sampling_rate_open_ephys = ephys_sampling_freq
    sampling_rate_rate = avg_sampling_rate_open_ephys/avg_sampling_rate_bonsai
    length = int(len(raw_sync_data) / sampling_rate_rate)
    indices = (np.arange(length) * sampling_rate_rate).astype(int)
    sync_data_ephys_downsampled = pd.DataFrame()
    sync_pulse = []
    print(indices)
    for index in indices:
        sync_pulse.append(raw_sync_data[index])
    sync_pulse = np.array(sync_pulse)[:,0]

    sync_data_ephys_downsampled = pd.DataFrame()
    sync_data_ephys_downsampled['sync_pulse'] = sync_pulse
    sync_data_ephys_downsampled['time'] = np.arange(0,len(sync_pulse))/avg_sampling_rate_bonsai

    return sync_data_ephys_downsampled


def synchronise_position_data_via_column_ttl_pulses(position_data, video_data, processed_path, recording_path):
    # get on and off times for ttl pulse
    position_data = get_behaviour_sync_on_and_off_times(position_data)
    video_data = get_video_sync_on_and_off_times(video_data)

    # calculate lag and align the position data
    video_data = adjust_for_lag(position_data, video_data, recording_path, processed_path)

    # remove negative time points
    video_data = video_data.reset_index(drop=True)
    del position_data["on_index_diff"]
    del position_data["on_index"]
    return position_data, video_data

def synchronise_position_data_via_ADC_ttl_pulses(position_data, processed_path, recording_path):
    # get downsampled ttl pulse from ephys data
    sync_data_ephys_downsampled = get_downsampled_ttl_pulse_array(recording_path, position_data)

    # get on and off times for ttl pulse
    position_data = get_video_sync_on_and_off_times(position_data)

    # calculate lag and align the position data
    position_data = adjust_for_lag_ephys(sync_data_ephys_downsampled, position_data, recording_path, processed_path)

    # remove negative time points
    position_data = position_data.drop(position_data[position_data.synced_time < 0].index)
    position_data = position_data.reset_index(drop=True)

    position_data["time_seconds"] = position_data["synced_time"]
    return position_data

