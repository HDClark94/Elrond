from run_pipeline import do_sorting_pipeline
from Elrond.Helpers.upload_download import get_chronologized_recording_paths
import sys

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]

raw_recording_paths = [get_chronologized_recording_paths(project_path, mouse, day)[0]]
do_sorting_pipeline(mouse, day, sorter_name, project_path, recording_paths = raw_recording_paths, deriv_path = project_path + f"derivatives/M{mouse}/D{day}/of1_only/")
