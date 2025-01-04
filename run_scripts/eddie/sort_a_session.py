from run_pipeline import do_sorting_pipeline
from Elrond.Helpers.upload_download import get_chronologized_recording_paths, get_session_names
import sys

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]
session_id = sys.argv[5]

raw_recording_paths = [get_chronologized_recording_paths(project_path, mouse, day)[int(session_id)]]
session_names = get_session_names(raw_recording_paths)

session_sorter_path = f"{project_path}derivatives/M{mouse}/D{day}/{session_names[0]}/{sorter_name}/session_sort/"

do_sorting_pipeline(mouse, day, sorter_name, project_path, recording_paths = raw_recording_paths, 
                    sorter_path = session_sorter_path + "sorting/",
                    sa_path =     session_sorter_path + "sa/",
                    report_path = session_sorter_path + "report/",
                    session_names=session_names
)
