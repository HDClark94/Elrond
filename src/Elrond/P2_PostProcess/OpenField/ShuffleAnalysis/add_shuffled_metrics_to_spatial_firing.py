import pandas as pd
import numpy as np
import os
import sys
import traceback
#import warnings
#warnings.filterwarnings("ignore")

def nan2val(scores, collumn):
    scores_without_nans = []
    for i in range(len(scores)):
        if np.isnan(scores[i]):
            if collumn == "speed_score":
                scores[i] = 0
            elif collumn == "hd_score":
                scores[i] = 0
            elif collumn == "rayleigh_score":
                scores[i] = 1
            elif collumn == "spatial_information_score":
                scores[i] = 0
            elif collumn == "grid_score":
                scores[i] = 0
            elif collumn == "border_score":
                scores[i] = 0
    return scores


def add_shuffled_cutoffs(recordings_folder_to_process):

    recording_list = [f.path for f in os.scandir(recordings_folder_to_process) if f.is_dir()]

    for recording_path in recording_list:
        print("processing ", recording_path)

        shuffle=pd.DataFrame()
        if os.path.isdir(recording_path+r"/MountainSort/DataFrames/shuffles"):
            shuffle_list = [f.path for f in os.scandir(recording_path+r"/MountainSort/DataFrames/shuffles") if f.is_file()]

            # remove shuffle.pkl if one is found, this is a deprecated pkl.
            if os.path.isfile(recording_path+r"/MountainSort/DataFrames/shuffles/shuffle.pkl"):
                shuffle_list.remove(recording_path+r"/MountainSort/DataFrames/shuffles/shuffle.pkl")

            for i in range(len(shuffle_list)):
                cluster_shuffle = pd.read_pickle(shuffle_list[i])
                shuffle = pd.concat([shuffle, cluster_shuffle], ignore_index=False)
            print("I have found a shuffled dataframe")

            if os.path.isfile(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl"):
                spatial_firing = pd.read_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

                if len(spatial_firing)>0:
                    print("cluster IDs in shuffle df: ", np.unique(shuffle.cluster_id))
                    print("cluster IDs in spatial df: ", np.unique(shuffle.cluster_id))

                    print("There are", len(shuffle)/len(spatial_firing), "shuffles per cell")

                    speed_threshold_poss = []
                    speed_threshold_negs = []
                    hd_thresholds = []
                    rayleigh_thresholds = []
                    spatial_thresholds = []
                    grid_thresholds = []
                    border_thresholds = []
                    half_session_thresholds = []

                    speed_n_nans_removed_from_shuffle = []
                    hd_n_nans_removed_from_shuffle = []
                    rayleigh_n_nans_removed_from_shuffle = []
                    spatial_n_nans_removed_from_shuffle = []
                    grid_n_nans_removed_from_shuffle = []
                    border_n_nans_removed_from_shuffle = []
                    half_session_n_nans_removed_from_shuffle = []

                    for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
                        cluster_shuffle_df = shuffle[(shuffle.cluster_id == cluster_id)] # dataframe for that cluster
                        print("For cluster", cluster_id, " there are ", len(cluster_shuffle_df), " shuffles")

                        speed_scores = np.array(cluster_shuffle_df["speed_score"])
                        hd_scores = np.array(cluster_shuffle_df["hd_score"])
                        rayleigh_scores = np.array(cluster_shuffle_df["rayleigh_score"])
                        spatial_information_scores = np.array(cluster_shuffle_df["spatial_information_score"])
                        grid_score = np.array(cluster_shuffle_df["grid_score"])
                        border_score = np.array(cluster_shuffle_df["border_score"])
                        half_session_score = np.array(cluster_shuffle_df["rate_map_correlation_first_vs_second_half"])

                        # count the number of nans from the shuffled distribution
                        speed_n_nans_removed_from_shuffle.append(len(cluster_shuffle_df)-np.count_nonzero(np.isnan(speed_scores)))
                        hd_n_nans_removed_from_shuffle.append(len(cluster_shuffle_df)-np.count_nonzero(np.isnan(hd_scores)))
                        rayleigh_n_nans_removed_from_shuffle.append(len(cluster_shuffle_df)-np.count_nonzero(np.isnan(rayleigh_scores)))
                        spatial_n_nans_removed_from_shuffle.append(len(cluster_shuffle_df)-np.count_nonzero(np.isnan(spatial_information_scores)))
                        grid_n_nans_removed_from_shuffle.append(len(cluster_shuffle_df)-np.count_nonzero(np.isnan(grid_score)))
                        border_n_nans_removed_from_shuffle.append(len(cluster_shuffle_df)-np.count_nonzero(np.isnan(border_score)))
                        half_session_n_nans_removed_from_shuffle.append(len(cluster_shuffle_df)-np.count_nonzero(np.isnan(half_session_score)))
                        # print it out for people to see
                        print("There are this many non-nan values for the grid score: ", grid_n_nans_removed_from_shuffle[cluster_index])
                        print("There are this many non-nan values for the border score: ", border_n_nans_removed_from_shuffle[cluster_index])
                        print("There are this many non-nan values for the half-session score: ", half_session_n_nans_removed_from_shuffle[cluster_index])
                        print("There are this many non-nan values for the spatial score: ", spatial_n_nans_removed_from_shuffle[cluster_index])
                        print("There are this many non-nan values for the hd score: ", hd_n_nans_removed_from_shuffle[cluster_index])
                        print("There are this many non-nan values for the speed score: ", speed_n_nans_removed_from_shuffle[cluster_index])

                        #remove the nan values
                        speed_scores = speed_scores[~np.isnan(speed_scores)]
                        hd_scores = hd_scores[~np.isnan(hd_scores)]
                        rayleigh_scores = rayleigh_scores[~np.isnan(rayleigh_scores)]
                        spatial_information_scores = spatial_information_scores[~np.isnan(spatial_information_scores)]
                        grid_score = grid_score[~np.isnan(grid_score)]
                        border_score = border_score[~np.isnan(border_score)]
                        half_session_score = half_session_score[~np.isnan(half_session_score)]

                        # calculate the 99th percentile threshold for individual clusters
                        # calculations based on z values please see https://sphweb.bumc.bu.edu/otlt/mph-modules/bs/bs704_probability/bs704_probability10.html
                        adjusted_speed_threshold_pos = np.nanmean(speed_scores) + (np.nanstd(speed_scores)*+1.960) # two tailed
                        adjusted_speed_threshold_neg = np.nanmean(speed_scores) + (np.nanstd(speed_scores)*-1.960) # two tailed
                        adjusted_hd_threshold = np.nanmean(hd_scores) + (np.nanstd(hd_scores)*2.326) # one tailed
                        adjusted_rayleigh_threshold = np.nanmean(rayleigh_scores) + (np.nanstd(rayleigh_scores)*-2.326) # one tailed
                        adjusted_spatial_threshold = np.nanmean(spatial_information_scores) + (np.nanstd(spatial_information_scores)*2.326) # one tailed
                        adjusted_grid_threshold = np.nanmean(grid_score) + (np.nanstd(grid_score)*2.326) # one tailed
                        adjusted_border_threshold = np.nanmean(border_score) + (np.nanstd(border_score)*2.326) # one tailed
                        adjusted_half_session_threshold = np.nanmean(half_session_score) + (np.nanstd(half_session_score)*2.326) # one tailed

                        speed_threshold_poss.append(adjusted_speed_threshold_pos)
                        speed_threshold_negs.append(adjusted_speed_threshold_neg)
                        hd_thresholds.append(adjusted_hd_threshold)
                        rayleigh_thresholds.append(adjusted_rayleigh_threshold)
                        spatial_thresholds.append(adjusted_spatial_threshold)
                        grid_thresholds.append(adjusted_grid_threshold)
                        border_thresholds.append(adjusted_border_threshold)
                        half_session_thresholds.append(adjusted_half_session_threshold)

                    spatial_firing["speed_threshold_pos"] = speed_threshold_poss
                    spatial_firing["speed_threshold_neg"] = speed_threshold_negs
                    spatial_firing["hd_threshold"] = hd_thresholds
                    spatial_firing["rayleigh_threshold"] = rayleigh_thresholds
                    spatial_firing["spatial_threshold"] = spatial_thresholds
                    spatial_firing["grid_threshold"] = grid_thresholds
                    spatial_firing["border_threshold"] = border_thresholds
                    spatial_firing["half_session_threshold"] = half_session_thresholds

                    spatial_firing["speed_n_nans_removed_from_shuffle"] = speed_n_nans_removed_from_shuffle
                    spatial_firing["hd_n_nans_removed_from_shuffle"] = hd_n_nans_removed_from_shuffle
                    spatial_firing["rayleigh_n_nans_removed_from_shuffle"] = rayleigh_n_nans_removed_from_shuffle
                    spatial_firing["spatial_n_nans_removed_from_shuffle"] = spatial_n_nans_removed_from_shuffle
                    spatial_firing["grid_n_nans_removed_from_shuffle"] = grid_n_nans_removed_from_shuffle
                    spatial_firing["border_n_nans_removed_from_shuffle"] = border_n_nans_removed_from_shuffle
                    spatial_firing["half_session_n_nans_removed_from_shuffle"] = half_session_n_nans_removed_from_shuffle

                    spatial_firing.to_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

                else:
                    print("There are no cells in this recordings")
            else:
                print("No spatial firing could be found")

def add_spatial_classifier_based_on_cutoffs(recordings_folder_to_process):
    recording_list = [f.path for f in os.scandir(recordings_folder_to_process) if f.is_dir()]

    for recording_path in recording_list:
        print("processing ", recording_path)
        if os.path.exists(recording_path+r"/MountainSort/DataFrames/shuffles/"):
            spatial_firing = pd.read_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

            grid_cells = []
            border_cells = []
            hd_cells = []
            hd_cells_rayleigh = []
            spatial_cells = []
            speed_cells = []

            for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
                cluster_spatial_firing = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

                if ((cluster_spatial_firing["grid_score"].iloc[0] > cluster_spatial_firing["grid_threshold"].iloc[0]) and
                    (cluster_spatial_firing["rate_map_correlation_first_vs_second_half"].iloc[0] > cluster_spatial_firing["half_session_threshold"].iloc[0])):
                    grid_cell = True
                else:
                    grid_cell = False

                if (((cluster_spatial_firing["border_score"].iloc[0] > 0.5) and
                     (cluster_spatial_firing["border_score"].iloc[0] > cluster_spatial_firing["border_threshold"].iloc[0]) and
                     (cluster_spatial_firing["rate_map_correlation_first_vs_second_half"].iloc[0] > cluster_spatial_firing["half_session_threshold"].iloc[0]))):
                    border_cell = True
                else:
                    border_cell = False

                if ((cluster_spatial_firing["hd_score"].iloc[0] > 0.2) and
                        (cluster_spatial_firing["hd_score"].iloc[0] > cluster_spatial_firing["hd_threshold"].iloc[0])):
                    hd_cell = True
                else:
                    hd_cell = False

                if (cluster_spatial_firing["rayleigh_score"].iloc[0] < cluster_spatial_firing["rayleigh_threshold"].iloc[0]):
                    hd_cell_rayleigh = True
                else:
                    hd_cell_rayleigh = False

                if ((cluster_spatial_firing["spatial_information_score"].iloc[0] > cluster_spatial_firing["spatial_threshold"].iloc[0]) and
                    (cluster_spatial_firing["rate_map_correlation_first_vs_second_half"].iloc[0] > cluster_spatial_firing["half_session_threshold"].iloc[0])):
                    spatial_cell = True
                else:
                    spatial_cell = False

                if (cluster_spatial_firing["speed_score"].iloc[0] > cluster_spatial_firing["speed_threshold_pos"].iloc[0]):
                    speed_cell = True
                elif (cluster_spatial_firing["speed_score"].iloc[0] < cluster_spatial_firing["speed_threshold_neg"].iloc[0]):
                    speed_cell = True
                else:
                    speed_cell = False

                grid_cells.append(grid_cell)
                border_cells.append(border_cell)
                hd_cells.append(hd_cell)
                hd_cells_rayleigh.append(hd_cell_rayleigh)
                spatial_cells.append(spatial_cell)
                speed_cells.append(speed_cell)

            spatial_firing["grid_cell"] = grid_cells
            spatial_firing["border_cell"] = border_cells
            spatial_firing["hd_cell"] = hd_cells
            spatial_firing["hd_cell_rayleigh"] = hd_cells_rayleigh
            spatial_firing["spatial_cell"] = spatial_cells
            spatial_firing["speed_cell"] = speed_cells

            spatial_firing.to_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

def add_spatial_classifier_based_on_classifiers(recordings_folder_to_process):
    recording_list = [f.path for f in os.scandir(recordings_folder_to_process) if f.is_dir()]

    for recording_path in recording_list:
        print("processing ", recording_path)
        if os.path.exists(recording_path+r"/MountainSort/DataFrames/shuffles/"):
            spatial_firing = pd.read_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

            classifier = []
            for cluster_index, cluster_id in enumerate(spatial_firing.cluster_id):
                cluster_spatial_firing = spatial_firing[(spatial_firing.cluster_id == cluster_id)] # dataframe for that cluster

                if (cluster_spatial_firing["grid_cell"].iloc[0] == 1):
                    classifier.append("G")
                elif (cluster_spatial_firing["border_cell"].iloc[0] == 1):
                    classifier.append("B")
                elif(cluster_spatial_firing["spatial_cell"].iloc[0] == 1):
                    classifier.append("NG")
                elif(cluster_spatial_firing["hd_cell"].iloc[0] == 1):
                    classifier.append("HD")
                else:
                    classifier.append("NS")

            spatial_firing["classifier"] = classifier
            spatial_firing.to_pickle(recording_path+r"/MountainSort/DataFrames/spatial_firing.pkl")

def main():
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('The shuffled analysis scripts (on Eddie) used python 3.8 to make the data frames. If you run this and '
          'get an error about pickle protocols, try to make a new python 3.8 virtual environment on Eleanor '
          '(conda create -n environmentname python=3.8) and use that. (The pipeline currently needs 3.6, so do not '
          'change that.')

    folders = []
    #folders.append("/mnt/datastore/Harry/Cohort9_Junji/of")
    #folders.append("/mnt/datastore/Harry/Cohort7_october2020/of")
    #folders.append("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort5/OpenField")
    #folders.append("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort4/OpenFeild")
    #folders.append("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort3/OpenFeild")
    #folders.append("/mnt/datastore/Sarah/Data/Ramp_project/OpenEphys/_cohort2/OpenField")
    #folders.append("/mnt/datastore/Harry/Cohort6_july2020/of")
    folders.append("/mnt/datastore/Harry/Cohort8_may2021/of")

    for folder in folders:
        add_shuffled_cutoffs(folder)
        add_spatial_classifier_based_on_cutoffs(folder)
        add_spatial_classifier_based_on_classifiers(folder)
    print("look now")

if __name__ == '__main__':
    main()