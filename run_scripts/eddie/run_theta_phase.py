from run_pipeline import do_theta_phase
from Elrond.Helpers.upload_download import get_chronologized_recording_paths
import sys

mouse = sys.argv[1]
day = sys.argv[2]
project_path = sys.argv[3]

raw_recording_paths = get_chronologized_recording_paths(project_path, mouse, day)
do_theta_phase(mouse, day, project_path, recording_paths = raw_recording_paths)
