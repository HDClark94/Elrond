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

zarr_job_name =  mouseday_string + "z_" + sorter_name
sort_job_name =  mouseday_string + sorter_name
sspp_job_name =  mouseday_string + "sspp_" + sorter_name

theta_job_name = mouseday_string + "theta"
of1_job_name = mouseday_string + "1dlc"
of2_job_name = mouseday_string + "2dlc"
behaviour_job_name = mouseday_string + "behave"
out_job_name = mouseday_string + "out_" + sorter_name

# Now run full pipeline on eddie

if len(paths_on_datastore) == 3:
        run_python_script(
        elrond_path + "/../../run_scripts/eddie/zarr_of1.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
        hold_jid = stagein_job_name + '_0,'+stagein_job_name + '_1,'+stagein_job_name + '_2',
        job_name = zarr_job_name + "of1",
        h_rt = "0:59:00"
        )
        run_python_script(
        elrond_path + "/../../run_scripts/eddie/zarr_of2.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
        hold_jid = stagein_job_name + '_0,'+stagein_job_name + '_1,'+stagein_job_name + '_2',
        job_name = zarr_job_name + "of2",
        h_rt = "0:59:00"
        )
        run_python_script(
        elrond_path + "/../../run_scripts/eddie/zarr_vr.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
        hold_jid = stagein_job_name + '_0,'+stagein_job_name + '_1,'+stagein_job_name + '_2',
        job_name = zarr_job_name,
        h_rt = "0:59:00"
        )
else:
    run_python_script(
        elrond_path + "/../../run_scripts/eddie/zarr_time.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
        hold_jid = stagein_job_name + '_0,'+stagein_job_name + '_1,'+stagein_job_name + '_2',
        job_name = zarr_job_name,
        h_rt = "0:59:00"
        )


run_python_script(
    elrond_path + "/../../run_scripts/eddie/sspp.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
    hold_jid = sort_job_name,
    job_name = sspp_job_name,
    h_rt = "1:59:00"
)


run_stageout_script({
    project_path + "derivatives/M"+mouse+"/D"+day+"/": "/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Chris/Cohort12/derivatives/M"+mouse+"/D"+day+"/"
    },
    hold_jid = behaviour_job_name,
    job_name = out_job_name
    )
