
import shutil
import os

def process_recordings(recording_paths, suffix, nwb_folder_name, destination_folder):
    for recording_path in recording_paths:
        nwb_file_names = [f for f in os.listdir(recording_path+nwb_folder_name) if f.endswith(suffix)]
        for nwb_file_name in nwb_file_names:
            shutil.copy(recording_path+nwb_folder_name+nwb_file_name, destination_folder+nwb_file_name)

def main():
    recording_paths = []
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/vr") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/vr") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/vr") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort8_may2021/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort7_october2020/of") if f.is_dir()])
    recording_paths.extend([f.path for f in os.scandir("/mnt/datastore/Harry/cohort6_july2020/of") if f.is_dir()])
    process_recordings(recording_paths,
                       suffix=".nwb",
                       nwb_folder_name="/processed/nwb/",
                       destination_folder="/mnt/datastore/Harry/dandiset/")

if __name__ == '__main__':
    main()