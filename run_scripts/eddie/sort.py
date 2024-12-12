from run_pipeline import do_just_sorting
from Elrond.Helpers.upload_download import get_chronologized_recording_paths, get_session_names
import sys
import json

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]
session_names = sys.argv[5]

session_names = json.load(session_names)

do_just_sorting(mouse, day, sorter_name, project_path, session_names=session_names)