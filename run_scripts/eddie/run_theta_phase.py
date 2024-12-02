from run_pipeline import do_theta_phase
import sys
from Elrond.Helpers.upload_download import get_chronologized_recording_paths, get_session_names

mouse = sys.argv[1]
day = sys.argv[2]
project_path = sys.argv[3]
session_index = sys.argv[4]

raw_recording_paths = [get_chronologized_recording_paths(project_path, mouse, day)[int(session_index)]]
session_names = get_session_names(raw_recording_paths)

do_theta_phase(mouse, day, project_path, raw_recording_paths, session_names)


