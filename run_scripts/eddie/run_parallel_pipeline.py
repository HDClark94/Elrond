import sys
import os
from Elrond.Helpers.create_eddie_scripts import stagein_data, run_python_script, run_stageout_script, run_gpu_python_script
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
    stagein_data(mouse, day, project_path, job_name = stagein_job_name, which_rec=0)
    if len(os.listdir(data_path)) > 2:
        stagein_data(mouse, day, project_path, job_name = stagein_job_name, which_rec=1)
        stagein_data(mouse, day, project_path, job_name = stagein_job_name, which_rec=2)

mouseday_string = "M" + mouse + "_" + day + "_"

zarr_job_name =  mouseday_string + "z_" + sorter_name
sort_job_name =  mouseday_string + sorter_name
sspp_job_name =  mouseday_string + "sspp_" + sorter_name

theta_job_name = mouseday_string + "theta"
of1_job_name = mouseday_string + "1dlc"
of2_job_name = mouseday_string + "2dlc"
behaviour_job_name = mouseday_string + "behave"

# Now run full pipeline on eddie

if len(os.listdir(data_path)) > 2:
        run_python_script(
        elrond_path + "/../../run_scripts/eddie/zarr_of1.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
        hold_jid = stagein_job_name,
        job_name = zarr_job_name + "of1",
        )
        run_python_script(
        elrond_path + "/../../run_scripts/eddie/zarr_of2.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
        hold_jid = stagein_job_name,
        job_name = zarr_job_name + "of2",
        )
        run_python_script(
        elrond_path + "/../../run_scripts/eddie/zarr_vr.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
        hold_jid = stagein_job_name,
        job_name = zarr_job_name,
        )
else:
    run_python_script(
        elrond_path + "/../../run_scripts/eddie/zarr_time.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
        hold_jid = stagein_job_name,
        job_name = zarr_job_name,
        )

if sorter_name == "kilosort4":
    run_gpu_python_script(
        elrond_path + "/../../run_scripts/eddie/sort.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
        hold_jid = zarr_job_name,
        job_name = sort_job_name,
        )
else:
    run_python_script(
        elrond_path + "/../../run_scripts/eddie/sort.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
        hold_jid = zarr_job_name,
        job_name = sort_job_name,
    )

run_python_script(
    elrond_path + "/../../run_scripts/eddie/sspp.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
    hold_jid = sort_job_name,
    job_name = sspp_job_name,
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
    hold_jid = sspp_job_name + "," + of1_job_name + "," + of2_job_name,
    job_name = behaviour_job_name,
    cores=3,
)

run_stageout_script({
    project_path + "derivatives/M"+mouse+"/D"+day+"/": "/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Chris/Cohort12/derivatives/M"+mouse+"/D"+day+"/"
    },
    hold_jid = behaviour_job_name,
    job_name = mouseday_string + "out_" + sorter_name
    )