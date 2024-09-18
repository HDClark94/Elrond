import pandas as pd
import os
import PostSorting.open_field_spatial_firing
import PostSorting.speed
import PostSorting.open_field_head_direction
import PostSorting.open_field_firing_maps
import PostSorting.open_field_grid_cells
import PostSorting.open_field_border_cells
import PostSorting.open_field_firing_fields
import PostSorting.compare_first_and_second_half
import numpy as np
import settings as settings
import time
from PostSorting import parameters

def run_parallel_of_shuffle(single_shuffle, synced_spatial_data):

    single_shuffle = PostSorting.open_field_spatial_firing.process_spatial_firing(single_shuffle, synced_spatial_data)
    single_shuffle = PostSorting.speed.calculate_speed_score(synced_spatial_data, single_shuffle, settings.gauss_sd_for_speed_score, settings.sampling_rate)
    _, single_shuffle = PostSorting.open_field_head_direction.process_hd_data(single_shuffle, synced_spatial_data)
    position_heatmap, single_shuffle = PostSorting.open_field_firing_maps.make_firing_field_maps(synced_spatial_data, single_shuffle)
    single_shuffle = PostSorting.open_field_grid_cells.process_grid_data(single_shuffle)
    single_shuffle = PostSorting.open_field_firing_maps.calculate_spatial_information(single_shuffle, position_heatmap)
    single_shuffle = PostSorting.open_field_border_cells.process_border_data(single_shuffle)
    single_shuffle, _, _, _, _ = PostSorting.compare_first_and_second_half.analyse_half_session_rate_maps(synced_spatial_data, single_shuffle)
    single_shuffle = single_shuffle[["cluster_id", "shuffle_id", "mean_firing_rate", "speed_score", "speed_score_p_values", "hd_score", "rayleigh_score",
                                     "spatial_information_score", "grid_score", "border_score", "rate_map_correlation_first_vs_second_half", "percent_excluded_bins_rate_map_correlation_first_vs_second_half_p"]]
    return single_shuffle

def generate_shuffled_times(cluster_firing, n_shuffles):
    cluster_firing = cluster_firing[["cluster_id", "firing_times", "mean_firing_rate", "recording_length_sampling_points"]]
    recording_length = int(cluster_firing["recording_length_sampling_points"].iloc[0])
    minimum_shift = int(20 * settings.sampling_rate)  # 20 seconds
    maximum_shift = int(recording_length - 20 * settings.sampling_rate)  # full length - 20 sec
    shuffle_firing = pd.DataFrame()
    for i in range(n_shuffles):
        shuffle = cluster_firing.copy()
        firing_times = shuffle["firing_times"].to_numpy()[0]
        random_firing_additions = np.random.randint(low=minimum_shift, high=maximum_shift)
        shuffled_firing_times = firing_times + random_firing_additions
        shuffled_firing_times[shuffled_firing_times >= recording_length] = shuffled_firing_times[shuffled_firing_times >= recording_length] - recording_length  # wrap around the firing times that exceed the length of the recording
        shuffle["firing_times"] = [shuffled_firing_times]
        shuffle_firing = pd.concat([shuffle_firing, shuffle], ignore_index=True)

    shuffle_firing["shuffle_id"] = np.arange(0, n_shuffles)
    return shuffle_firing

def one_job_shuffle_parallel(recording_path, cluster_id, n_shuffles):
    '''
    creates a single shuffle of each cell and saves it in recording/sorter/dataframes/shuffles/
    :param recording_path: path to a recording folder
    :param shuffle_id: integer id of a single shuffle
    '''
    time0 = time.time()
    checkpoint_interval = 30*60 # in seconds
    checkpoint_counter = 1

    spike_data_spatial = pd.read_pickle(recording_path+"/MountainSort/DataFrames/spatial_firing.pkl")
    synced_spatial_data = pd.read_pickle(recording_path+"/MountainSort/DataFrames/position.pkl")
    cluster_spike_data = spike_data_spatial[(spike_data_spatial["cluster_id"] == cluster_id)]

    if os.path.isfile(recording_path + "/MountainSort/DataFrames/shuffles/"+str(int(cluster_id))+"_shuffle.pkl"):
        shuffle = pd.read_pickle(recording_path + "/MountainSort/DataFrames/shuffles/"+str(int(cluster_id))+"_shuffle.pkl")
        n_shuffles_pre_computed = len(shuffle)
    else:
        shuffle = pd.DataFrame()
        n_shuffles_pre_computed = 0

    shuffles_to_run = n_shuffles-n_shuffles_pre_computed

    if shuffles_to_run > 1:
        for i in range(shuffles_to_run):
            if len(cluster_spike_data["firing_times"]) == 0:
                shuffled_cluster_spike_data = pd.DataFrame(np.nan, index=[0], columns=["cluster_id", "shuffle_id",
                                                                                       "mean_firing_rate", "speed_score",
                                                                                       "speed_score_p_values", "hd_score",
                                                                                       "rayleigh_score",
                                                                                       "spatial_information_score",
                                                                                       "grid_score", "border_score",
                                                                                       "rate_map_correlation_first_vs_second_half",
                                                                                       "percent_excluded_bins_rate_map_correlation_first_vs_second_half_p"])

            else:
                shuffled_cluster_spike_data = generate_shuffled_times(cluster_spike_data, n_shuffles=1)
                shuffled_cluster_spike_data = run_parallel_of_shuffle(shuffled_cluster_spike_data, synced_spatial_data)

            shuffle = pd.concat([shuffle, shuffled_cluster_spike_data], ignore_index=True)
            print(i, " shuffle complete")

            time_elapsed = time.time()-time0

            if time_elapsed > (checkpoint_interval*checkpoint_counter):
                checkpoint_counter += 1
                checkpoint(shuffle, cluster_id, recording_path)

        checkpoint(shuffle, cluster_id, recording_path)
    print("shuffle analysis completed for ", recording_path)
    return

def checkpoint(shuffle, cluster_id, recording_path):
    if not os.path.exists(recording_path+"/MountainSort/DataFrames/shuffles"):
        os.mkdir(recording_path+"/MountainSort/DataFrames/shuffles")
    shuffle.to_pickle(recording_path + "/MountainSort/DataFrames/shuffles/"+str(int(cluster_id))+"_shuffle.pkl")
    print("checkpoint saved")

def run_shuffle_analysis_vr(recording, n_shuffles):
    return

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

    #========================FOR RUNNING ON FROM TERMINAL=====================================#
    #=========================================================================================#
    recording_path = os.environ['RECORDING_PATH']
    n_shuffles = int(os.environ['SHUFFLE_NUMBER'])
    cluster_id = int(os.environ["CLUSTER_ID"])
    one_job_shuffle_parallel(recording_path, cluster_id, n_shuffles)

    #================FOR RUNNING ON ELEANOR (SINGLE RECORDING)================================#
    #=========================================================================================#

    #recording_path = '/mnt/datastore/Harry/cohort8_may2021/of/M11_D21_2021-06-07_09-46-58'
    #spatial_firing = pd.read_pickle(recording_path+"/MountainSort/DataFrames/spatial_firing.pkl")
    #for cluster_id in spatial_firing["cluster_id"]:
    #    one_job_shuffle_parallel(recording_path, cluster_id, n_shuffles=1000)

    #=========================================================================================#
    #=========================================================================================#

if __name__ == '__main__':
    main()