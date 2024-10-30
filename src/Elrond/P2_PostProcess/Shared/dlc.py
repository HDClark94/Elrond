
import pandas as pd
import os
from pathlib import Path
import shutil

def extract_from_dlc(video_path, save_path, model_path):
    
    import deeplabcut as dlc
    
    dlc_data = pd.DataFrame()

    config_path = model_path + "config.yaml"

    Path(save_path).mkdir(exist_ok=True)

    # add columns to position_data using markers as set in the dlc model
    video_filename = video_path.split("/")[-1]
    new_videopath = save_path + video_filename

    _ = shutil.copyfile(video_path, new_videopath)
    dlc.analyze_videos(config_path, [new_videopath], save_as_csv=True, destfolder=save_path)
    dlc.filterpredictions(config_path, [new_videopath])
    dlc.create_labeled_video(config_path, [new_videopath], save_frames=False)
    dlc.plot_trajectories(config_path, [new_videopath])
    csv_path = [os.path.abspath(os.path.join(save_path, filename)) for filename in os.listdir(save_path) if filename.endswith(".csv")]
    dlc_data = pd.read_csv(csv_path[0], header=[1, 2], index_col=0)  # ignore the scorer column
    os.remove(new_videopath)

    return dlc_data