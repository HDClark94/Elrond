import sys
import os
from Elrond.Helpers.create_eddie_scripts import stagein_data, run_python_script, run_stageout_script, run_gpu_python_script
from pathlib import Path
import Elrond
import yaml
import time
from Elrond.Helpers.upload_download import get_session_names


def get_filepaths_on_datastore(mouse, day, project_path):

    elrond_path = Elrond.__path__[0]
    run_python_script(
        python_arg = f"{elrond_path}/../../run_scripts/eddie/eddie_get_filenames.py {mouse} {day} {project_path}", 
        venv="elrond", 
        staging=True, 
        h_rt="0:29:59", 
        cores=1,
        job_name=f"M{mouse}_{day}_getfilenames",
    )

    return 


mice_days_string = sys.argv[1]
sorter_name = sys.argv[2]
project_path = sys.argv[3]

mice_days = yaml.load(mice_days_string, Loader=yaml.Loader)

elrond_path = Elrond.__path__[0]

for mouse, days in mice_days.items():
    mouse = str(mouse)
    for day in days:
        day = str(day)

        data_path = project_path + f"data/M{mouse}_D{day}/"
        Path(data_path).mkdir(exist_ok=True)

        # check if raw recordings are on eddie. If not, stage them
        paths_on_datastore = []
        filenames_path = data_path + "data_folder_names.txt"
        
        if Path(filenames_path).exists() is False:
            get_filepaths_on_datastore(mouse, day, project_path)

        while Path(filenames_path).exists() is False:
            time.sleep(5)

        with open(filenames_path) as f:
            paths_on_datastore = f.read().splitlines()

        session_names = get_session_names(paths_on_datastore)
        print(f"Sessions for M{mouse} D{day} are: {session_names}")
            
        mouseday_string = "M" + mouse + "_" + day + "_"
        stagein_job_name = mouseday_string + "in"

        if len(os.listdir(data_path)) == 1:
            stagein_data(mouse, day, project_path, job_name = stagein_job_name + "_" + str(0), which_rec=0, hold_jid="sleepy_time")
            

            for a, path in enumerate(paths_on_datastore):
                if a == 0:
                    continue
                else:
                    stagein_data(mouse, day, project_path, job_name = stagein_job_name + "_" + str(a), which_rec=a, hold_jid="sleepy_time")

        for session_id, session_name in enumerate(session_names):

            zarr_job_name =  mouseday_string + "z_" + session_name + sorter_name 
            sort_job_name =  mouseday_string + session_name + sorter_name 

            theta_job_name = mouseday_string + "theta" + session_name
            of1_job_name = mouseday_string + "1dlc" + session_name
            of2_job_name = mouseday_string + "2dlc" + session_name
            behaviour_job_name = mouseday_string + "behave" + session_name
            out_job_name = mouseday_string + "out_" + sorter_name + session_name

            run_python_script(
                elrond_path + "/../../run_scripts/eddie/sort_a_session.py " + mouse + " " + day + " " + sorter_name + " " + project_path + " " + session_id, 
                hold_jid = stagein_job_name + '_0,'+stagein_job_name + '_1,'+stagein_job_name + '_2',
                job_name = sort_job_name,
                h_rt = "11:59:59"
            )

            if 'of1' in session_names:
                run_python_script(
                    elrond_path + "/../../run_scripts/eddie/dlc_of1.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
                    hold_jid = stagein_job_name + "_0",
                    job_name = of1_job_name,
                    h_rt = "1:59:00"
                )

            if 'of2' in session_names:
                run_python_script(
                    elrond_path + "/../../run_scripts/eddie/dlc_of2.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
                    hold_jid = stagein_job_name+ "_2",
                    job_name = of2_job_name,
                    h_rt = "1:59:00"
                )

            # Run behaviour, once everything else is done
            run_python_script(
                elrond_path + "/../../run_scripts/eddie/behaviour.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
                hold_jid = sort_job_name + "," + of1_job_name + "," + of2_job_name,
                job_name = behaviour_job_name,
                cores=3,
                h_rt = "1:59:00"
            )

            run_stageout_script({
                project_path + "derivatives/M"+mouse+"/D"+day+"/": "/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Chris/Cohort12/derivatives/M"+mouse+"/D"+day+"/"
                },
                hold_jid = behaviour_job_name,
                job_name = out_job_name
            )


