import sys
import os
from Elrond.Helpers.create_eddie_scripts import stagein_data, run_python_script, run_stageout_script, run_gpu_python_script, get_filepaths_on_datastore
from Elrond.Helpers.upload_download import get_session_names, chronologize_paths
from pathlib import Path
import Elrond

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mouse", help="Mouse number, e.g. 20", type=int)
parser.add_argument("day", help="Day number, e.g. 14", type=int)
parser.add_argument("sorter_name", help="Sorter preset e.g. 'kilosort4' or 'kilosort4_motion_correction'. Current presets can be found in P1_SpikeSort/defaults.py")
parser.add_argument("project_path", help="Folder containing 'data' and 'derivative' folders")
parser.add_argument("--session_names", help="List of sessions you'd like to process e.g. ['of1', 'vr']. If not given, processes all sessions.")
parser.add_argument("--just_sort", help="Bool. Just do the sorting stuff: no dlc, no theta phase.", type=bool)

args = parser.parse_args()

mouse = args.mouse
day = args.day
sorter_name = args.sorter_name
project_path = args.project_path

if args.session_names:
    session_names = args.session_names
else:
    session_names = None

if args.just_sort:
    just_sort = args.just_sort
else:
    just_sort = False

print(f"Using {sorter_name} to sort mouse {mouse}, day {day}")

elrond_path = Elrond.__path__[0]

data_path = project_path + f"data/M{mouse}_D{day}"
Path(data_path).mkdir(exist_ok=True)

# Get the filenames from datastore.
filenames_path = project_path + f"data/M{mouse}_D{day}/data_folder_names.txt"

if Path(filenames_path).exists() is False:
    get_filepaths_on_datastore(mouse, day, project_path)

while Path(filenames_path).exists() is False:
    time.sleep(5)

paths_on_datastore = []
with open(filenames_path) as f:
    paths_on_datastore = f.read().splitlines()

all_session_names = get_session_names(chronologize_paths(paths_on_datastore))

if session_names is not None:
    which_sessions = [session_name in session_names for session_name in all_session_names]
    print(which_sessions)
    session_names = all_session_names[which_sessions]
    paths_on_datastore = paths_on_datastore[which_sessions]
    print(session_names)

print(f"Doing sessions: {session_names}")

stagein_job_name = f"stagein_M{mouse}_D{day}_"

for a, (session_name, path_on_datastore) in enumerate(zip(session_names,paths_on_datastore)):
    stagein_data(mouse, day, project_path, path_on_datastore, job_name = stagein_job_name + session_name)

stagein_job_names = ""
for session_name in session_names:
    stagein_job_names += stagein_job_name + session_name + ","
stagein_job_names = stagein_job_names[:-1]

mouseday_string = "M" + str(mouse) + "_" + str(day) + "_"

zarr_job_name =  mouseday_string + "z_" + sorter_name
sort_job_name =  mouseday_string + sorter_name
sspp_job_name =  mouseday_string + "sspp_" + sorter_name

theta_job_name = mouseday_string + "theta"
location_job_name = mouseday_string + "loc"
of1_job_name = mouseday_string + "1dlc"
of2_job_name = mouseday_string + "2dlc"
behaviour_job_name = mouseday_string + "behave"
out_job_name = mouseday_string + "out_" + sorter_name

# Now run full pipeline on eddie

for session_name in session_names:

    run_python_script(
        elrond_path + "/../../run_scripts/eddie/zarr_time.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
        hold_jid = stagein_job_name + session_name,
        job_name = zarr_job_name + session_name,
        h_rt = "0:59:00"
    )

zarr_job_names = ""
for session_name in session_names:
    zarr_job_names += zarr_job_name + session_name + ","
zarr_job_names = zarr_job_names[:-1]



# if sorter_name == "kilosort4":
#     run_gpu_python_script(
#         elrond_path + "/../../run_scripts/eddie/sort.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
#         hold_jid = zarr_job_name,
#         job_name = sort_job_name,
#         )
# else:
run_python_script(
    elrond_path + "/../../run_scripts/eddie/sort.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
    hold_jid = zarr_job_names,
    job_name = sort_job_name,
    h_rt = "23:59:59"
)

run_python_script(
    elrond_path + "/../../run_scripts/eddie/sspp.py " + mouse + " " + day + " " + sorter_name + " " + project_path,
    hold_jid = sort_job_name,
    job_name = sspp_job_name,
    h_rt = "3:59:00",
    cores=1
)

if just_sort is False:

    # Run location plots
    run_python_script(
        elrond_path + "/../../run_scripts/eddie/location_plots.py " + mouse + " " + day + " " + sorter_name + " " + project_path,
        hold_jid = zarr_job_names,
        job_name = location_job_name,
        cores=4,
    )

    # Run theta phase
    for a, session_name in enumerate(session_names):
        # Run theta phase
        run_python_script(
            elrond_path + "/../../run_scripts/eddie/run_theta_phase.py " + mouse + " " + day + " " + project_path + " " + str(a),
            hold_jid = stagein_job_names,
            job_name = theta_job_name + "_" + session_name,
            cores=4,
        )

    # Run DLC on of1
    if 'of1' in session_names:
        run_python_script(
            elrond_path + "/../../run_scripts/eddie/dlc_of1.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
            hold_jid = stagein_job_names,
            job_name = of1_job_name,
            h_rt = "1:59:00"
            )

    # Run DLC on of2
    if 'of2' in session_names:
        run_python_script(
            elrond_path + "/../../run_scripts/eddie/dlc_of2.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
            hold_jid = stagein_job_names,
            job_name = of2_job_name,
            h_rt = "1:59:00"
            )

# Run behaviour, once everything else is done
run_python_script(
    elrond_path + "/../../run_scripts/eddie/behaviour.py " + mouse + " " + day + " " + sorter_name + " " + project_path, 
    hold_jid = sspp_job_name + "," + of1_job_name + "," + of2_job_name,
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
