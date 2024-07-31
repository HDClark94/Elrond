import numpy as np
import os
import pandas as pd
import pickle
import Elrond.settings as settings
import PreClustering.dead_channels
import PostSorting.compare_first_and_second_half
import PostSorting.curation
import PostSorting.lfp
import PostSorting.load_firing_data
import PostSorting.load_snippet_data
import PostSorting.make_opto_plots
import PostSorting.make_plots
import PostSorting.open_field_border_cells
import PostSorting.open_field_firing_fields
import PostSorting.open_field_firing_maps
import PostSorting.open_field_grid_cells
import PostSorting.open_field_head_direction
import PostSorting.open_field_light_data
import PostSorting.open_field_make_plots
import PostSorting.open_field_spatial_data
import PostSorting.open_field_spatial_firing
import PostSorting.open_field_sync_data
import PostSorting.parameters
import PostSorting.speed
import PostSorting.temporal_firing
import PostSorting.theta_modulation
import PostSorting.load_snippet_data_opto
from Elrond.Helpers import file_utility, open_ephys_IO
import sys
import traceback
prm = PostSorting.parameters.Parameters()


def create_folders_for_output(recording_to_process):
    if os.path.exists(recording_to_process + '/Figures') is False:
        os.makedirs(recording_to_process + '/Figures')
    if os.path.exists(recording_to_process + '/DataFrames') is False:
        os.makedirs(recording_to_process + '/DataFrames')


def initialize_parameters(recording_to_process):
    prm.set_pixel_ratio(440)
    prm.set_opto_channel('100_ADC3.continuous')
    if os.path.exists(recording_to_process + '/100_ADC1.continuous'):
        prm.set_sync_channel('100_ADC1.continuous')
    elif os.path.exists(recording_to_process + '/105_CH20_2_0.continuous'):
        prm.set_sync_channel('105_CH20_2_0.continuous')
    else:
        prm.set_sync_channel('105_CH20_0.continuous')

    prm.set_ephys_channels(PostSorting.load_firing_data.available_ephys_channels(recording_to_process, prm))
    prm.set_sampling_rate(30000)
    prm.set_local_recording_folder_path(recording_to_process)
    prm.set_file_path(recording_to_process)
    prm.set_ms_tmp_path('/tmp/mountainlab/')


# process tags from parameters.txt metadata file. These are in the third line of the file.
def process_running_parameter_tag(running_parameter_tags):
    unexpected_tag = False
    pixel_ratio = False

    if not running_parameter_tags:
        return unexpected_tag, pixel_ratio

    tags = [x.strip() for x in running_parameter_tags.split('*')]
    for tag in tags:
        if tag.startswith('pixel_ratio'):
            pixel_ratio = int(tag.split('=')[1])  # put pixel ratio value in pixel_ratio
            prm.set_pixel_ratio(pixel_ratio)
        else:
            print('Unexpected / incorrect tag in the third line of parameters file: ' + str(unexpected_tag))
            unexpected_tag = True

    if pixel_ratio is False:
        print('Default pixel ratio (440) is used.')
    return unexpected_tag, pixel_ratio

# check for opto pulses and make opto dataframe if found
def process_light_stimulation(recording_to_process, opto_channel, output_path):
    opto_on, opto_off, is_found, start_idx, end_idx = PostSorting.open_field_light_data.process_opto_data(recording_to_process, opto_channel)

    if is_found:
        opto_data_frame = PostSorting.open_field_light_data.make_opto_data_frame(opto_on)
        if os.path.exists(output_path + '/DataFrames') is False:
            os.makedirs(output_path + '/DataFrames')
        opto_data_frame.to_pickle(output_path + '/DataFrames/opto_pulses.pkl')

    return opto_on, opto_off, is_found, start_idx, end_idx


# removes data points from before first pulse and after last pulse with 1s buffer on either side
def remove_exploration_without_opto(start_idx, end_idx, spatial_data, sampling_rate):
    if start_idx is None:
        return spatial_data

    else:
        opto_start_s, opto_end_s = int(start_idx/sampling_rate) - 1, int(end_idx/sampling_rate) + 1  # index of first and last pulse
        nearest_index_opto_start = (np.abs(spatial_data.synced_time - opto_start_s)).argmin()  # nearest bonsai index to start
        nearest_index_opto_end = (np.abs(spatial_data.synced_time - opto_end_s)).argmin()  # nearest bonsai index to end
        spatial_data.drop(range(0, nearest_index_opto_start), inplace=True)  # drop data on either side
        spatial_data.drop(range(nearest_index_opto_end, spatial_data.index[-1]), inplace=True)
        spatial_data = spatial_data.iloc[:-1, :]  # drop last row
        length_seconds = spatial_data.synced_time.values[-1] - spatial_data.synced_time.values[0]  # new recording length

    return spatial_data, length_seconds


# remove firing times from before and after start of opto stimulation
def remove_spikes_without_opto(spike_data, spatial_firing, sampling_rate):
    spikes_during_opto = []
    opto_start = int((spatial_firing.synced_time.values[0]) * sampling_rate)  # in sampling points
    opto_end = int((spatial_firing.synced_time.values[-1]) * sampling_rate)  # in sampling points
    recording_length = opto_end - opto_start  # in sampling points

    for cluster_id, cluster in spike_data.iterrows():
        firing_times_all = cluster.firing_times
        firing_times_from_opto_start = np.take(firing_times_all, np.where(firing_times_all >= opto_start)[0])
        firing_times_during_opto = np.take(firing_times_from_opto_start, np.where(firing_times_from_opto_start <= opto_end)[0])
        spikes_during_opto.append(firing_times_during_opto)

    spike_data['firing_times'] = spikes_during_opto  # replace with firing times during opto
    spike_data['firing_times_opto'] = spikes_during_opto  # this col name is required for opto analysis scripts
    spike_data['recording_length_sampling_points'] = int(recording_length)  # replace with opto recording length

    return spike_data


def save_data_frames(spatial_firing, synced_spatial_data=None, snippet_data=None, bad_clusters=None, lfp_data=None):
    print('I will save the data frames now.')
    if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
        os.makedirs(prm.get_output_path() + '/DataFrames')
    spatial_firing.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing.pkl')
    if synced_spatial_data is not None:
        synced_spatial_data.to_pickle(prm.get_output_path() + '/DataFrames/position.pkl')
    if snippet_data is not None:
        snippet_data.to_pickle(prm.get_output_path() + '/DataFrames/snippet_data.pkl')
    if bad_clusters is not None:
        bad_clusters.to_pickle(prm.get_output_path() + '/DataFrames/noisy_clusters.pkl')
    if lfp_data is not None:
        lfp_data.to_pickle(prm.get_output_path() + "/DataFrames/lfp_data.pkl")


def make_openfield_plots(position_data, spatial_firing, position_heat_map, hd_histogram, output_path, prm):
    PostSorting.make_plots.plot_waveforms(spatial_firing, output_path)
    PostSorting.make_plots.plot_spike_histogram(spatial_firing, output_path)
    PostSorting.make_plots.plot_firing_rate_vs_speed(spatial_firing, position_data, prm)
    PostSorting.make_plots.plot_speed_vs_firing_rate(position_data, spatial_firing, prm.get_sampling_rate(), 250, prm)
    PostSorting.make_plots.plot_autocorrelograms(spatial_firing, output_path)
    PostSorting.open_field_make_plots.plot_spikes_on_trajectory(position_data, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_coverage(position_heat_map, prm)
    PostSorting.open_field_make_plots.plot_firing_rate_maps(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_rate_map_autocorrelogram(spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_hd(spatial_firing, position_data, prm)
    PostSorting.open_field_make_plots.plot_polar_head_direction_histogram(hd_histogram, spatial_firing, prm)
    PostSorting.open_field_make_plots.plot_hd_for_firing_fields(spatial_firing, position_data, prm)
    PostSorting.open_field_make_plots.plot_spikes_on_firing_fields(spatial_firing, prm)
    PostSorting.open_field_make_plots.make_combined_figure(prm, spatial_firing)


def save_data_for_plots(position_heat_map, hd_histogram, prm):
    if os.path.exists(prm.get_output_path() + '/DataFrames') is False:
        os.makedirs(prm.get_output_path() + '/DataFrames')
    np.save(prm.get_output_path() + '/DataFrames/position_heat_map.npy', position_heat_map)
    np.save(prm.get_output_path() + '/DataFrames/hd_histogram.npy', hd_histogram)
    file_handler = open(prm.get_output_path() + '/DataFrames/prm', 'wb')
    pickle.dump(prm, file_handler)


# find recording length when there is no position data
def set_recording_length(recording_to_process, prm):
    is_found, total_length = False, None
    print('I am loading a channel to find out the length of the recording, because there is no position data available.')
    file_path = recording_to_process + '/' + prm.get_sync_channel()
    if os.path.exists(file_path):
        continuous_channel_data = open_ephys_IO.get_data_continuous(file_path)
        total_length = len(continuous_channel_data) / settings.sampling_rate  # convert to seconds
        is_found = True
    else:
        print('I could not load the channel and set the recording length.')
    return total_length, is_found


def analyze_snippets_and_temporal_firing(recording_to_process, prm, dead_channels, opto_start_index, total_length, segment_id=0):
    """
    Run analyses on spike sorted data to analyze snippets and temporal firing properties.
    """
    number_of_channels, _ = file_utility.count_files_that_match_in_folder(recording_to_process, data_file_prefix=settings.data_file_prefix, data_file_suffix='.continuous')
    spike_data = PostSorting.load_firing_data.process_firing_times(recording_to_process, prm.sorter_name, dead_channels, opto_tagging_start_index=opto_start_index, segment_id=segment_id)
    spike_data = PostSorting.temporal_firing.add_temporal_firing_properties_to_df(spike_data, total_length, number_of_channels)
    spike_data, bad_clusters = PostSorting.curation.curate_data(spike_data, prm.sorter_name,prm.get_local_recording_folder_path(),prm.get_ms_tmp_path())
    spike_data = PostSorting.load_snippet_data.get_snippets(spike_data, recording_to_process, prm.sorter_name,dead_channels, random_snippets=False, segment_id=segment_id)
    snippet_data = PostSorting.load_snippet_data.get_snippets(spike_data, recording_to_process, prm.sorter_name, dead_channels, random_snippets=True, segment_id=segment_id)

    return spike_data, snippet_data, bad_clusters


def make_opto_plots(spatial_firing, prm):
    output_path = prm.get_output_path()
    PostSorting.make_plots.plot_waveforms(spatial_firing, output_path)
    PostSorting.make_plots.plot_spike_histogram(spatial_firing, output_path)
    PostSorting.make_plots.plot_autocorrelograms(spatial_firing, output_path)
    PostSorting.make_opto_plots.make_optogenetics_plots(spatial_firing, output_path, prm.get_sampling_rate())


# analyse subset of pulses and save to subfolder: 'first_pulses' or 'last_pulses'
def analyse_subset_of_pulses(spatial_firing, prm, pulses, window_fs, opto_output_path, segment_id):
    prm.set_output_path(opto_output_path)
    if os.path.exists(opto_output_path + '/DataFrames') is False:
        os.makedirs(opto_output_path + '/DataFrames')
    pulses.to_pickle(opto_output_path + '/DataFrames/opto_pulses.pkl')  # save copy of opto pulses to subfolder
    spatial_firing = PostSorting.open_field_light_data.process_spikes_around_light(spatial_firing, prm, subset=True, pulses=pulses, window_fs=window_fs, segment_id=segment_id)
    spatial_firing.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing_opto.pkl')  # save copy with opto stats
    make_opto_plots(spatial_firing, prm)


def process_first_and_last_spikes(spatial_firing, window_ms, prm, segment_id=0, num_pulses=200, threshold=2000):
    output_path, sampling_rate = prm.get_output_path(), prm.get_sampling_rate()
    opto_pulses = pd.read_pickle(output_path + '/DataFrames/opto_pulses.pkl')
    on_pulses = opto_pulses.opto_start_times

    if len(on_pulses) > threshold:  # run if > 2000 opto pulses
        print("I will now analyse the first and last", num_pulses, "opto pulses separately.")
        total_num_pulses = len(on_pulses)
        window_size_sampling_rate = int(sampling_rate / 1000 * window_ms)
        first_pulses, last_pulses = on_pulses[0:num_pulses], on_pulses[(total_num_pulses-num_pulses):total_num_pulses]
        first_output_path, last_output_path = output_path + "/first_pulses", output_path + "/last_pulses"
        analyse_subset_of_pulses(spatial_firing, prm, first_pulses, window_size_sampling_rate, first_output_path, segment_id)
        analyse_subset_of_pulses(spatial_firing, prm, last_pulses, window_size_sampling_rate, last_output_path, segment_id)


def analyse_opto_data(opto_on, spatial_firing, prm, segment_id=0):
    """
    :param opto_on: times of where opto pulse is on
    :param spatial_firing: spike data
    This analysis occurs independently from analysis of spatial firing data
    """
    window = PostSorting.open_field_light_data.find_stimulation_frequency(opto_on, prm.sampling_rate)  # calculate analysis window
    print('I will now process the peristimulus spikes. This will take a while for high frequency stimulations...')
    spatial_firing = PostSorting.open_field_light_data.process_spikes_around_light(spatial_firing, prm, window_size_ms=window, segment_id=segment_id)
    spatial_firing.to_pickle(prm.get_output_path() + '/DataFrames/spatial_firing_opto.pkl')  # save copy with opto stats
    make_opto_plots(spatial_firing, prm)  # make plots
    process_first_and_last_spikes(spatial_firing, window, prm, segment_id)  # separately analyse first/last pulses (if >1000 pulses


def process_opto_with_position(recording, spatial_data, lfp_data, opto_found, opto_on, start_idx, end_idx, prm, dead_channels, output_path, segment_id=0):
    """
    Analyses sessions where opto-stimulation happens during open field exploration.
    Position and spatial firing data is analysed from the start of the first pulse to the end of the last pulse + a 1s buffer.
    """
    synced_spatial_data, recording_length, is_found = PostSorting.open_field_sync_data.process_sync_data(recording, prm, spatial_data)

    spike_data = PostSorting.load_firing_data.process_firing_times(recording, prm.sorter_name, dead_channels, opto_tagging_start_index=None, segment_id=segment_id)

    if opto_found:  # remove position data and spikes before and after stimulation period
        synced_spatial_data, recording_length = remove_exploration_without_opto(start_idx, end_idx, synced_spatial_data, prm.sampling_rate)
        spike_data = remove_spikes_without_opto(spike_data, synced_spatial_data, prm.sampling_rate)

    # add temporal firing properties and curate clusters
    number_of_channels, _ = file_utility.count_files_that_match_in_folder(recording, data_file_prefix=settings.data_file_prefix, data_file_suffix='.continuous')
    spike_data = PostSorting.temporal_firing.add_temporal_firing_properties_to_df(spike_data, recording_length, number_of_channels)
    spike_data, bad_clusters = PostSorting.curation.curate_data(spike_data, prm.sorter_name, prm.get_local_recording_folder_path(), prm.get_ms_tmp_path())

    if len(spike_data) == 0:  # this means that there are no good clusters and the analysis will not run
        save_data_frames(spike_data, synced_spatial_data, bad_clusters=bad_clusters, lfp_data=lfp_data)

    else:  # process position data and output as normal for open-field trials
        snippet_data = PostSorting.load_snippet_data.get_snippets(spike_data, recording, prm.sorter_name, dead_channels, random_snippets=True, segment_id=segment_id)
        spike_data_spatial = PostSorting.open_field_spatial_firing.process_spatial_firing(spike_data, synced_spatial_data)
        spike_data_spatial = PostSorting.speed.calculate_speed_score(synced_spatial_data, spike_data_spatial, settings.gauss_sd_for_speed_score, settings.sampling_rate)
        hd_histogram, spatial_firing = PostSorting.open_field_head_direction.process_hd_data(spike_data_spatial, synced_spatial_data)
        position_heat_map, spatial_firing = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, spike_data_spatial)
        spatial_firing = PostSorting.open_field_firing_maps.calculate_spatial_information(spatial_firing, position_heat_map)
        spatial_firing = PostSorting.open_field_grid_cells.process_grid_data(spatial_firing)
        spatial_firing = PostSorting.open_field_firing_fields.analyze_firing_fields(spatial_firing, synced_spatial_data, prm)
        spatial_firing = PostSorting.open_field_border_cells.process_border_data(spatial_firing)
        spatial_firing = PostSorting.theta_modulation.calculate_theta_index(spatial_firing, output_path, settings.sampling_rate)
        spatial_firing, spike_data_first, spike_data_second, synced_spatial_data_first, synced_spatial_data_second = PostSorting.compare_first_and_second_half.analyse_half_session_rate_maps(synced_spatial_data, spatial_firing)
        make_openfield_plots(synced_spatial_data, spatial_firing, position_heat_map, hd_histogram, output_path, prm)
        PostSorting.open_field_make_plots.make_combined_field_analysis_figures(prm, spatial_firing)
        save_data_frames(spatial_firing, synced_spatial_data, snippet_data=snippet_data, lfp_data=lfp_data)
        save_data_for_plots(position_heat_map, hd_histogram, prm)

        if opto_found:  # then analyse opto data, if found
            analyse_opto_data(opto_on, spatial_firing, prm, segment_id=segment_id)
    return

def process_optotagging(recording, prm, opto_found, opto_on, start_idx, dead_channels, segment_id=0):
    if opto_found:
        total_length, is_found = set_recording_length(recording, prm)
        spike_data, snippet_data, bad_clusters = analyze_snippets_and_temporal_firing(recording, prm, dead_channels, start_idx, total_length, segment_id=segment_id)

        if len(spike_data) > 0:  # only runs if there are curated clusters
            spike_data = PostSorting.theta_modulation.calculate_theta_index(spike_data, prm.get_output_path(), settings.sampling_rate)
            analyse_opto_data(opto_on, spike_data, prm, segment_id=segment_id)
            save_data_frames(spike_data, snippet_data=snippet_data, bad_clusters=bad_clusters)
    else:
        print('There were no opto pulses for this recording. Opto analysis will not run.')


def post_process_recording(recording_to_process, session_type, running_parameter_tags=False, sorter_name=settings.sorterName, segment_id=0):
    """
    Analyses data from opto-stimulation sessions during behaviour (session_type = openfield_opto) or without animal
    position (session_type = opto, ie optotagging after behaviour). If position data cannot be processed, analysis will
    still run for opto stimulation.
    """
    create_folders_for_output(recording_to_process)
    initialize_parameters(recording_to_process)
    unexpected_tag, pixel_ratio = process_running_parameter_tag(running_parameter_tags)
    prm.set_sorter_name('/' + sorter_name)
    prm.set_output_path(recording_to_process + prm.get_sorter_name())
    output_path = recording_to_process + '/' + settings.sorterName
    PreClustering.dead_channels.get_dead_channel_ids(prm)
    dead_channels = prm.get_dead_channels()
    ephys_channels = prm.get_ephys_channels()
    output_path = recording_to_process + '/' + settings.sorterName
    opto_channel = prm.get_opto_channel()

    if pixel_ratio is False:
        print('Default pixel ratio (440) is used.')
    else:
        prm.set_pixel_ratio(pixel_ratio)

    lfp_data = PostSorting.lfp.process_lfp(recording_to_process, ephys_channels, output_path, dead_channels)

    spatial_data, position_is_found = PostSorting.open_field_spatial_data.process_position_data(recording_to_process, prm, do_resample=False)

    # check for opto, get on and off times and start/end indices for opto
    opto_on, opto_off, opto_is_found, start, end = process_light_stimulation(recording_to_process, opto_channel, output_path)

    if (session_type == 'openfield_opto') and position_is_found:
        try:
            process_opto_with_position(recording_to_process, spatial_data, lfp_data, opto_is_found, opto_on, start, end,
                                       prm, dead_channels, output_path, segment_id=segment_id)

        except Exception as ex:  # analyse opto data only if there is an error with the position data
            print('There is a problem with this file. '
                  'I will move on to the next one. This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)

            print('I cannot analyze the position data for this opto recording, I will run the opto analysis only.')
            process_optotagging(recording_to_process, prm, opto_is_found, opto_on, start, dead_channels=dead_channels,segment_id=0)

    else:  # if problem with position file, process opto without position
        process_optotagging(recording_to_process, prm, opto_is_found, opto_on, start, dead_channels=dead_channels, segment_id=0)
