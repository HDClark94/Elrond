import pandas as pd
import settings
import traceback
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from neuroconv.utils.dict import load_dict_from_file

def analyse_licking(recording, video_path, update_spatial_firing, update_processed_position_data):
    return

def get_eye_bounds(recording):
    # look for a bounding_box_parameter_file
    # if one isn't found, use a default set of coordinates
    try:
        if os.path.exists(recording.split("/vr/")[0]+"/parameter_helper.csv"):
            parameter_helper = pd.read_csv(recording.split("/vr/")[0]+"/parameter_helper.csv")
            session_id = recording.split("/")[-1]
            mouse_id = get_mouse(session_id)
            training_day = get_day(session_id)
            parameter_helper["mouse_id"]= parameter_helper["mouse_id"].astype(str)
            parameter_helper_mouse_day = parameter_helper[(parameter_helper.mouse_id == mouse_id) &
                                                          (parameter_helper.training_day == training_day)]
            anterior_x = parameter_helper_mouse_day["eye_anterior_x"].iloc[0]
            anterior_y = parameter_helper_mouse_day["eye_anterior_y"].iloc[0]
            posterior_x = parameter_helper_mouse_day["eye_posterior_x"].iloc[0]
            posterior_y = parameter_helper_mouse_day["eye_posterior_y"].iloc[0]
    except:
        anterior_x = 950
        anterior_y = 150
        posterior_x = 150
        posterior_y = 120

    return anterior_x, anterior_y, posterior_x, posterior_y


def get_eye_bounding_box_pixel_coordinates(recording):
    # look for a bounding_box_parameter_file
    # if one isn't found, use a default set of coordinates
    try:
        if os.path.exists(recording.split("/vr/")[0]+"/parameter_helper.csv"):
            parameter_helper = pd.read_csv(recording.split("/vr/")[0]+"/parameter_helper.csv")
            session_id = recording.split("/")[-1]
            mouse_id = get_mouse(session_id)
            training_day = get_day(session_id)
            parameter_helper["mouse_id"]= parameter_helper["mouse_id"].astype(str)
            parameter_helper_mouse_day = parameter_helper[(parameter_helper.mouse_id == mouse_id) &
                                                          (parameter_helper.training_day == training_day)]
            eye_bounding_box_x_origin = parameter_helper_mouse_day["eye_bounding_box_x_origin"].iloc[0]
            eye_bounding_box_x_length = parameter_helper_mouse_day["eye_bounding_box_x_length"].iloc[0]
            eye_bounding_box_y_origin = parameter_helper_mouse_day["eye_bounding_box_y_origin"].iloc[0]
            eye_bounding_box_y_length = parameter_helper_mouse_day["eye_bounding_box_y_length"].iloc[0]
    except:
        eye_bounding_box_x_origin = 950
        eye_bounding_box_x_length = 150
        eye_bounding_box_y_origin = 150
        eye_bounding_box_y_length = 120

    upper_left_x = eye_bounding_box_x_origin
    upper_right_x = eye_bounding_box_x_origin+eye_bounding_box_x_length
    lower_right_x = eye_bounding_box_x_origin+eye_bounding_box_x_length
    lower_left_x = eye_bounding_box_x_origin
    upper_left_y = eye_bounding_box_y_origin
    upper_right_y = eye_bounding_box_y_origin
    lower_right_y = eye_bounding_box_y_origin + eye_bounding_box_y_length
    lower_left_y = eye_bounding_box_y_origin + eye_bounding_box_y_length

    return upper_left_x, upper_right_x, lower_right_x, lower_left_x,\
        upper_left_y, upper_right_y, lower_right_y, lower_left_y




def analyse_pupils(recording, video_path, update_spatial_firing, update_processed_position_data):
    save_path = recording + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    upper_left_x, upper_right_x, lower_right_x, lower_left_x,\
        upper_left_y, upper_right_y, lower_right_y, lower_left_y = get_eye_bounding_box_pixel_coordinates(recording)

    anterior_x, anterior_y, posterior_x, posterior_y = get_eye_bounds(recording)

    print("I am attempted to analyse pupil dilation")

    vidcap = cv2.VideoCapture(recording+"/"+video_path)

    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    frame_id = int(length/2) # look at the middle frame by default to avoid any edge artefacts

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = vidcap.read()
    greyscale_frame = np.nanmean(frame, axis=2)
    fig, ax = plt.subplots()
    ax.imshow(frame)
    ax.plot([upper_left_x, upper_right_x, lower_right_x, lower_left_x, upper_left_x],
            [upper_left_y, upper_right_y, lower_right_y, lower_left_y, upper_left_y], linewidth=3, color='red')
    ax.scatter(anterior_x, anterior_y, marker="x", color="red")
    ax.scatter(posterior_x, posterior_y, marker="x", color="red")
    ax.plot([anterior_x, posterior_x], [anterior_y, posterior_y], linewidth=0.5, color='red')
    plt.savefig(save_path+"/middle_frame.png")
    plt.close()

    bounding_box = frame[upper_left_y:lower_right_y, upper_left_x:upper_right_x]
    bounding_box_gs = greyscale_frame[upper_left_y:lower_right_y, upper_left_x:upper_right_x]
    fig, ax = plt.subplots()
    ax.imshow(bounding_box)
    plt.savefig(save_path+"/bounding_box.png")
    plt.close()

    fig, ax = plt.subplots()
    gray = cv2.cvtColor(bounding_box, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    ax.imshow(blurred)
    plt.savefig(save_path+"/blurred.png")
    plt.close()

    fig, ax = plt.subplots()
    gray = cv2.cvtColor(bounding_box, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    ax.imshow(blurred, cmap="Greys")
    plt.savefig(save_path+"/greyblurred.png")
    plt.close()

    fig, ax = plt.subplots()
    thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    ax.imshow(thresh)
    plt.savefig(save_path + "/thresh.png")
    plt.close()

    fig, ax = plt.subplots()
    (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ax.imshow(threshInv)
    plt.savefig(save_path + "/OTSU.png")
    plt.close()

    # Read image
    im = cv2.imread("blob.jpg", cv2.IMREAD_GRAYSCALE)
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector()
    # Detect blobs.
    keypoints = detector.detect(im)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector()
    # Detect blobs.
    keypoints = detector.detect(frame)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax.imshow(im_with_keypoints)
    plt.savefig(save_path + "/im_with_keypoints.png")
    plt.close()

    x_test = np.linspace(anterior_x, posterior_x, 100)
    y_test = np.linspace(anterior_y, posterior_y, 100)
    slice_intensity = []
    for x, y in zip(x_test, y_test):
        closest_pixel_x = int(np.round(x))
        closest_pixel_y = int(np.round(y))
        slice_intensity.append(greyscale_frame[closest_pixel_y, closest_pixel_x])
    slice_intensity = np.array(slice_intensity)
    fig, ax = plt.subplots()
    ax.plot(slice_intensity)
    plt.savefig(save_path + "/slice_intensity.png")
    plt.close()
    return


def process_recordings(vr_recording_path_list):
    for recording in vr_recording_path_list:
        print("processing ", recording)
        try:
            # look for a video to analyse
            all_files_names = [f for f in os.listdir(recording) if os.path.isfile(os.path.join(recording, f))]
            all_video_file_names = [s for s in all_files_names if s.endswith(".avi")]

            if len(all_video_file_names) == 0:
                print("There are no videos in this recording to process")
            elif len(all_video_file_names) > 1:
                print("There are more than 1 video in this recording. There should only be one")
            else:
                print("There is a video in this recording. I will try process it")
                analyse_licking(recording, video_path=all_video_file_names[0], update_spatial_firing=False, update_processed_position_data=False)
                analyse_pupils(recording, video_path=all_video_file_names[0], update_spatial_firing=False, update_processed_position_data=False)
                print("successfully processed on "+recording)

        except Exception as ex:
            print('This is what Python says happened:')
            print(ex)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback)
            print("couldn't process vr_grid analysis on "+recording)


def get_bounding_box(x_origin, x_length, y_origin, y_length):
    upper_left_x = x_origin
    upper_right_x = x_origin+x_length
    lower_right_x = x_origin+x_length
    lower_left_x = x_origin
    upper_left_y = y_origin
    upper_right_y = y_origin
    lower_right_y = y_origin + y_length
    lower_left_y = y_origin + y_length
    return upper_left_x, upper_right_x, lower_right_x, lower_left_x,\
        upper_left_y, upper_right_y, lower_right_y, lower_left_y


def plot_led(params, video_path, save_path):
    upper_left_x, upper_right_x, lower_right_x, lower_left_x, \
    upper_left_y, upper_right_y, lower_right_y, lower_left_y = \
        get_bounding_box(params["vr_sync_pulse_led_bounding_box_x_origin"], params["vr_sync_pulse_led_bounding_box_x_length"],
                         params["vr_sync_pulse_led_bounding_box_y_origin"], params["vr_sync_pulse_led_bounding_box_y_length"])
    vidcap = cv2.VideoCapture(video_path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_id = int(length/2) # look at the middle frame by default to avoid any edge artefacts
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = vidcap.read()
    greyscale_frame = np.nanmean(frame, axis=2)
    bounding_box = frame[upper_left_y:lower_right_y, upper_left_x:upper_right_x]

    fig, ax = plt.subplots()
    ax.imshow(frame)
    plt.title("middle frame")
    plt.savefig(save_path + "/middle_frame.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.imshow(frame)
    ax.plot([upper_left_x, upper_right_x, lower_right_x, lower_left_x, upper_left_x],
            [upper_left_y, upper_right_y, lower_right_y, lower_left_y, upper_left_y], linewidth=3, color='red')
    plt.title("middle frame with led bounding box")
    plt.savefig(save_path + "/middle_frame_with_bounding_box.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.imshow(bounding_box)
    plt.title("led bounding box")
    plt.savefig(save_path + "/bounding_box.png")
    plt.close()


def process_video(recording_path, processed_folder_name, position_data):
    # look for video in recording_path, use deep lab cut to analyse on cpu frame by frame,
    # align with position_data using sync pulse from led in the video
    # look for a video to analyse
    # get bounding box for led
    if os.path.exists(recording_path+"/params.yml"):
        params = load_dict_from_file(recording_path+"/params.yml")
    else:
        raise AssertionError("I need a params file in order to process video")

    save_path = recording_path+"/"+processed_folder_name+"/video"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    video_paths = [os.path.abspath(os.path.join(recording_path, filename))
                   for filename in os.listdir(recording_path) if (filename.endswith(".mp4") or filename.endswith(".avi"))]
    if len(video_paths) != 1:
        raise AssertionError("I only want one video filename!")
    plot_led(params, video_paths[0], save_path) # only take the first video as there hopefully is only one

    # add lick or no lick and pupil radius to position data using deeplabcut model
    #add_licks(position_data, video_path=video_filenames[0])
    #add_pupil_radius(position_data, video_path=video_filenames[0])
    return position_data

#  for testing
def main():
    print('-------------------------------------------------------------')

    vr_path_list = ["/mnt/datastore/Harry/Cohort10_october2023/vr/M18_D27_2023-12-08_15-35-35"]
    process_recordings(vr_path_list)
    print('---------------video processing finished---------------------')
    print('-------------------------------------------------------------')

if __name__ == '__main__':
    main()