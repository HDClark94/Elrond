import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from neuroconv.utils.dict import load_dict_from_file
import settings
import deeplabcut as dlc

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

def analyse_dlc_model(position_data, video_path, model_path):
    config_path = model_path+"/config.yaml"
    # add columns to position_data using markers as set in the dlc model
    dlc.analyze_videos(config_path, [video_path], save_as_csv=True)

    return position_data


def process_video(recording_path, processed_folder_name, position_data):
    # run checks
    avi_paths = [os.path.abspath(os.path.join(recording_path, filename)) for filename in os.listdir(recording_path) if filename.endswith("_capture.avi")]
    csv_paths = [os.path.abspath(os.path.join(recording_path, filename)) for filename in os.listdir(recording_path) if filename.endswith("_capture.csv")]
    if (len(avi_paths) != 1) or (len(csv_paths) != 1):
        print("I couldn't process video because I need exactly 1 .avi file and 1 .csv file in the recording folder")
        return position_data
    if os.path.exists(recording_path+"/params.yml"):
        params = load_dict_from_file(recording_path+"/params.yml")
    else:
        raise AssertionError("I need a params file in order to process video")

    save_path = recording_path+"/"+processed_folder_name+"/video"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    position_data = analyse_dlc_model(position_data, video_path=avi_paths[0], model_path=settings.deeplabcut_project_path)

    # then process however you want
    plot_middle_frame(avi_paths[0], save_path) # only take the first video as there hopefully is only one
    #add_licks(position_data, video_path=video_filenames[0])
    #add_pupil_radius(position_data, video_path=video_filenames[0])
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