import sys
import os
from Elrond.Helpers.create_eddie_scripts import stagein_data, run_python_script, run_stageout_script, run_gpu_python_script
from Elrond.Helpers.upload_download import get_chronologized_recording_paths, get_session_names
from pathlib import Path
import Elrond

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]

elrond_path = Elrond.__path__[0]

data_path = project_path + f"data/M{mouse}_D{day}"
Path(data_path).mkdir(exist_ok=True)

# check if raw recordings are on eddie. If not, stage them
paths_on_datastore = []
stagein_job_name = None

filenames_path = project_path + f"data/M{mouse}_D{day}/data_folder_names.txt"
    
stagein_job_name = f"stagein_M{mouse}_D{day}"

if len(os.listdir(data_path)) < 2:
    stagein_data(mouse, day, project_path, job_name = stagein_job_name + "_" + str(0), which_rec=0)
    with open(filenames_path) as f:
        paths_on_datastore = f.read().splitlines()

    for a, path in enumerate(paths_on_datastore):
        if a == 0:
            continue
        else:
            stagein_data(mouse, day, project_path, job_name = stagein_job_name + "_" + str(a), which_rec=a)

mouseday_string = "M" + mouse + "_" + day + "_"
theta_job_name = mouseday_string + "theta"
out_job_name = mouseday_string + "out"

import time
time.sleep(60)


raw_recording_paths = get_chronologized_recording_paths(project_path, mouse, day)
session_names = get_session_names(raw_recording_paths)

for raw_recording_path, session_name in zip(raw_recording_paths, session_names):
    # Run theta phase
    run_python_script(
        elrond_path + "/../../run_scripts/eddie/run_theta_phase.py " + mouse + " " + day + " " + project_path + " " + raw_recording_path + " " + session_name,
        hold_jid = stagein_job_name + "_0,"+stagein_job_name + "_1,"+stagein_job_name + "_2",
        job_name = theta_job_name + session_name,
        cores=3,
    )

run_stageout_script({
    project_path + "derivatives/M"+mouse+"/D"+day+"/": "/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Chris/Cohort12/derivatives/M"+mouse+"/D"+day+"/"
    },
    hold_jid = theta_job_name + "vr,"+theta_job_name + "of1,"+theta_job_name + "of2",
    job_name = out_job_name
)
