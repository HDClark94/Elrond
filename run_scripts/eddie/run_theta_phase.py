from run_pipeline import do_theta_phase
import sys

mouse = sys.argv[1]
day = sys.argv[2]
project_path = sys.argv[3]
raw_recording_path = sys.argv[4]
session_name = sys.argv[5]

raw_recording_paths = [raw_recording_path]
session_names = [session_name]

do_theta_phase(mouse, day, project_path, raw_recording_paths, session_names)


