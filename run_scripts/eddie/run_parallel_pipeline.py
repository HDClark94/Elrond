import sys
import os
from Elrond.Helpers.create_eddie_scripts import stagein_data, run_python_script, run_stageout_script
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
stagein_job_name = None
if len(os.listdir(data_path)) < 3:
    stagein_job_name = f"stagein_M{mouse}_D{day}"
    stagein_data(mouse, day, project_path, job_name = stagein_job_name)

mouseday_string = "M" + mouse + "_" + day + "_"

pipeline_job_name = mouseday_string + sorter_name + "_pipe_full"
theta_job_name = mouseday_string + "theta"
of1_job_name = mouseday_string + "1dlc"
of2_job_name = mouseday_string + "2dlc"
behaviour_job_name = mouseday_string + "behave"

# Now run full pipeline on eddie
run_python_script(
    elrond_path + "/../../run_scripts/eddie/sorting.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
    hold_jid = stagein_job_name,
    job_name = pipeline_job_name,
    )

# Run theta phase
run_python_script(
    elrond_path + "/../../run_scripts/eddie/run_theta_phase.py " + mouse + " " + day + " " + project_path,
    hold_jid = stagein_job_name,
    job_name = theta_job_name,
    cores=3,
    )

# Run DLC on of1
run_python_script(
    elrond_path + "/../../run_scripts/eddie/dlc_of1.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
    hold_jid = stagein_job_name,
    job_name = of1_job_name,
    )

# Run DLC on of2
run_python_script(
    elrond_path + "/../../run_scripts/eddie/dlc_of2.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
    hold_jid = stagein_job_name,
    job_name = of2_job_name,
    )

# Run behaviour, once everything else is done
run_python_script(
    elrond_path + "/../../run_scripts/eddie/behaviour.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
    hold_jid = pipeline_job_name + "," + of1_job_name + "," + of2_job_name,
    job_name = behaviour_job_name,
    cores=3,
)

run_stageout_script({
    project_path + "derivatives/M"+mouse+"/D"+day+"/": "/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Chris/Cohort12/derivatives/M"+mouse+"/D"+day+"/"
    },
    hold_jid = behaviour_job_name,
    job_name = mouseday_string + "out_" + sorter_name
    )
