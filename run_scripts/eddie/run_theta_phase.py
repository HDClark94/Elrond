from run_pipeline import do_theta_phase
from Elrond.Helpers.upload_download import get_chronologized_recording_paths, get_session_names
import sys

mouse = sys.argv[1]
day = sys.argv[2]
project_path = sys.argv[3]

raw_recording_paths = get_chronologized_recording_paths(project_path, mouse, day)
session_names = get_session_names(raw_recording_paths)

for raw_path, session_name in zip(raw_recording_paths, session_names):
    save_path = project_path + f"derivatives/M{mouse}/D{day}/{session_name}/"
    do_theta_phase(raw_path, save_path)


