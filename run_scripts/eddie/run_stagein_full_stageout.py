import sys
import os
from Elrond.Helpers.create_eddie_scripts import stagein_data, run_python_script, run_stageout_script
from pathlib import Path

mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]

import Elrond
elrond_path = Elrond.__path__[0]

data_path = project_path + f"data/M{mouse}_D{day}"
Path(data_path).mkdir(exist_ok=True)

# check if raw recordings are on eddie. If not, stage them
stagein_job_name = None
if len(os.listdir(data_path)) < 3:
    stagein_job_name = f"M{mouse}_{day}_in"
    stagein_data(mouse, day, project_path, job_name = stagein_job_name, run=False)

pipeline_job_name = "M" + mouse + "_" + day + "_" + sorter_name + "_pipe_full"

# Now run full pipeline on eddie
run_python_script(
    elrond_path + "/../../run_scripts/run_pipeline.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
    hold_jid = stagein_job_name,
    job_name = pipeline_job_name, 
    run=False
    )

run_stageout_script({
    project_path + "derivatives/M"+mouse+"/D"+day+"/": "/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Chris/Cohort12/derivatives/M"+mouse+"/D"+day+"/"
    },
    hold_jid = pipeline_job_name,
    job_name = "M" + mouse + "_" + day + "_out_" + sorter_name,
    run=False
    )
