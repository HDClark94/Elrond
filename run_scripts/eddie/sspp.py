from run_pipeline import do_spikesort_postprocessing
from Elrond.Helpers.upload_download import get_chronologized_recording_paths, get_session_names
import sys

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]
session_names = sys.argv[5]

session_names = eval(session_names)

do_spikesort_postprocessing(mouse, day, sorter_name, project_path, session_names=session_names)