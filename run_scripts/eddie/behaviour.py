from run_pipeline import do_behavioural_postprocessing
from Elrond.Helpers.upload_download import get_chronologized_recording_paths, get_session_names
import sys

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]

raw_recording_paths = get_chronologized_recording_paths(project_path, mouse, day)
session_names = get_session_names(raw_recording_paths)
do_behavioural_postprocessing(mouse, day, sorter_name, project_path, recording_paths = raw_recording_paths, session_names=session_names)