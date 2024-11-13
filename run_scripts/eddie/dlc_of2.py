from run_pipeline import do_dlc_pipeline
from Elrond.Helpers.upload_download import get_chronologized_recording_paths
import sys

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]

raw_recording_paths = get_chronologized_recording_paths(project_path, mouse, day)
do_dlc_pipeline(mouse, day, project_path, dlc_of_model_path = project_path + "derivatives/dlc/of_cohort12-krs-2024-10-30/", do_of1=False, do_vr=False, recording_paths = raw_recording_paths)

