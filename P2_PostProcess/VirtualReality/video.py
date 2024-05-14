import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from neuroconv.utils.dict import load_dict_from_file
import settings
import deeplabcut as dlc
import numpy as np
import shutil
import math
from P2_PostProcess.VirtualReality.spatial_data import *
from P2_PostProcess.Shared.time_sync import *

def get_distance(x1,y1,x2,y2):
  distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
  return distance

def plot_middle_frame(video_path, save_path):
    vidcap = cv2.VideoCapture(video_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = int(length/2) # look at the middle frame by default to avoid any edge artefacts
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = vidcap.read()

    fig, ax = plt.subplots()
    ax.imshow(frame)
    plt.title("middle frame")
    plt.savefig(save_path + "/middle_frame.png")
    plt.close()

def analyse_dlc_model(video_path, model_path, save_path):
    config_path = model_path+"/config.yaml"
    # add columns to position_data using markers as set in the dlc model
    video_filename = video_path.split("/")[-1]
    new_videopath = save_path+"/"+video_filename
    _ = shutil.copy(video_path, new_videopath)

    dlc.analyze_videos(config_path, [new_videopath], save_as_csv=True, destfolder=save_path)
    dlc.filterpredictions(config_path, [new_videopath])
    dlc.create_labeled_video(config_path, [new_videopath], save_frames=False)
    dlc.plot_trajectories(config_path, [new_videopath])
    csv_path = [os.path.abspath(os.path.join(save_path, filename)) for filename in os.listdir(save_path) if filename.endswith(".csv")]
    markers_df = pd.read_csv(csv_path[0], header=[1, 2], index_col=0) # ignore the scorer column
    os.remove(new_videopath)
    return markers_df

def add_eye_stats(markers):
    radi = []
    centroids = []
    for i in range(len(markers)): # first two rows are names of bodyparts and coords
        centroid = np.mean(np.array([[markers[('eye_n', 'x')][i], markers[('eye_n', 'y')][i]],
                                     [markers[('eye_ne', 'x')][i], markers[('eye_ne', 'y')][i]],
                                     [markers[('eye_e', 'x')][i], markers[('eye_e', 'y')][i]],
                                     [markers[('eye_se', 'x')][i], markers[('eye_se', 'y')][i]],
                                     [markers[('eye_s', 'x')][i], markers[('eye_s', 'y')][i]],
                                     [markers[('eye_sw', 'x')][i], markers[('eye_sw', 'y')][i]],
                                     [markers[('eye_w', 'x')][i], markers[('eye_w', 'y')][i]],
                                     [markers[('eye_nw', 'x')][i], markers[('eye_nw', 'y')][i]]]), axis=0)
        radius = np.mean([get_distance(markers[('eye_n', 'x')][i], markers[('eye_n', 'y')][i], markers[('eye_s', 'x')][i], markers[('eye_s', 'y')][i]),
                             get_distance(markers[('eye_e', 'x')][i], markers[('eye_e', 'y')][i], markers[('eye_w', 'x')][i], markers[('eye_w', 'y')][i]),
                             get_distance(markers[('eye_ne', 'x')][i], markers[('eye_ne', 'y')][i], markers[('eye_sw', 'x')][i], markers[('eye_sw', 'y')][i]),
                             get_distance(markers[('eye_nw', 'x')][i], markers[('eye_nw', 'y')][i], markers[('eye_se', 'x')][i], markers[('eye_se', 'y')][i])])/2
        radi.append(radius)
        centroids.append(centroid)
    markers["eye_radius"] = radi
    markers["eye_centroid"] = centroids
    return markers

def add_synced_videodata_to_position_data(position_data, video_data):
    # at this point the position "time_seconds" and video data "synced_time"
    # are within the same time reference frame

    eye_radi = []
    eye_centroids_x = []
    eye_centroids_y = []
    for i in range(len(position_data)):
        time_second = position_data["time_seconds"][i]
        # get the closest synced time from the video_data
        absolute_diffs = np.abs(video_data['synced_time'] - time_second)
        closest_index = np.argmin(absolute_diffs)
        # put variables of interest into the position_dataframe
        eye_radi.append(video_data['eye_radius'][closest_index])
        eye_centroids_x.append(video_data['eye_centroid'][closest_index][0])
        eye_centroids_y.append(video_data['eye_centroid'][closest_index][1])
    position_data["eye_radius"] = eye_radi
    position_data["eye_centroid_x"] = eye_centroids_x
    position_data["eye_centroid_y"] = eye_centroids_y
    return position_data

def process_video(recording_path, processed_folder_name, position_data):
    # run checks
    avi_paths = [os.path.abspath(os.path.join(recording_path, filename)) for filename in os.listdir(recording_path) if filename.endswith("_capture.avi")]
    bonsai_csv_paths = [os.path.abspath(os.path.join(recording_path, filename)) for filename in os.listdir(recording_path) if filename.endswith("_capture.csv")]
    if (len(avi_paths) != 1) or (len(bonsai_csv_paths) != 1):
        print("I couldn't process video because I need exactly 1 .avi file and 1 .csv file in the recording folder")
        return position_data
    # else continue on with the video analysis

    save_path = recording_path+"/"+processed_folder_name+"/video"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # use bonsai csv to sync the video_markers_df with position_data and the sync pulse signal in that
    bonsai_data = read_bonsai_file(bonsai_csv_paths[0])
    video_data = analyse_dlc_model(video_path=avi_paths[0], model_path=settings.vr_deeplabcut_project_path, save_path=save_path)
    video_data = pd.concat([video_data.reset_index(drop=True), bonsai_data.reset_index(drop=True)], axis=1)
    video_data = add_eye_stats(video_data)

    # syncrhonise position data and video data
    position_data, video_data = synchronise_position_data_via_column_ttl_pulses(position_data, video_data)
    # now video data contains a synced_time column which is relative to the start of the time column in position_data
    position_data = add_synced_videodata_to_position_data(position_data, video_data)

    plot_middle_frame(avi_paths[0], save_path) # only take the first video as there hopefully is only one
    return position_data

#  for testing
def main():
    print('-------------------------------------------------------------')
    _ = process_video(recording_path="/mnt/datastore/Harry/Cohort11_april2024/vr/M21_D3_2024-04-26_09-16-13",
                      processed_folder_name="processed",
                      position_data=pd.DataFrame())
    print('---------------video processing finished---------------------')
    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()