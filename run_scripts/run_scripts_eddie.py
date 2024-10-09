import os
import sys
import subprocess

mouse = sys.argv[1]
day = sys.argv[2]
print("mouse: ", mouse)
print("day: ", day)

scratch_path = "/exports/eddie/scratch/hclark3/"
project_path = scratch_path + "harry_project/"
data_path = project_path + "data/M" + mouse + "_D" + day + "/"
code_path = project_path + "code/M" + mouse + "/D" + day + "/"
ders_path = project_path + "derivatives/M" + mouse + "/D" + day + "/"

os.makedirs(code_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)
os.makedirs(ders_path, exist_ok=True)

active_projects_path = "/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/"
source_path = active_projects_path + "Harry/Cohort11_april2024/"
output_path = source_path + "derivatives/M" + mouse + "/D" + day + "/"
source_path_DLC_OF_MODEL = active_projects_path+"Harry/deeplabcut/openfield_pose_eddie"
source_path_DLC_VR_LICK_MODEL = active_projects_path+"Harry/deeplabcut/Mouse_Licks-Harry_Clark-2024-09-10_eddie"
source_path_DLC_VR_PUPIL_MODEL = active_projects_path+"Harry/deeplabcut/vr-hc-2024-03-14_eddie"
email = "hclark3@ed.ac.uk"
repo_path = "/home/hclark3/Elrond/"

def staging_file_text(mouse, day):
    mouse_day_dir = "M"+mouse + "_" +"D"+ day

    return """#!/bin/sh
# Grid Engine options can go here (always start with #$ )
# Name job and set to use current working directory
#$ -cwd
#$ -N M"""+mouse+"""_D"""+day+"""_DL

# Choose the staging queue
#$ -q staging

# Hard runtime limit
#$ -l h_rt=01:00:00

# Source path on DataStore. It should start with one of
# /exports/csce/datastore, /exports/chss/datastore, /exports/cmvm/datastore or /exports/igmm/datastore
SOURCE=""" + source_path + """

# Destination path on Eddie. It should start with one of:
# /exports/csce/eddie, /exports/chss/eddie, /exports/cmvm/eddie, /exports/igmm/eddie or /exports/eddie/scratch
DESTINATION=""" + data_path + """

# Do the copy with rsync (avoid -p or -a options)
rsync -rm --exclude='*/*/processed/' --include='of/' --include='of/"""+mouse_day_dir+"""*' --include='of/"""+mouse_day_dir+"""*/****' --include='vr/' --include='vr/"""+mouse_day_dir+"""*' --include='vr/"""+mouse_day_dir+"""*/****' --exclude="*" ${SOURCE} ${DESTINATION}

# Source path on DataStore. It should start with one of
# /exports/csce/datastore, /exports/chss/datastore, /exports/cmvm/datastore or /exports/igmm/datastore
SOURCE=""" + source_path_DLC_OF_MODEL + """

# Destination path on Eddie. It should start with one of:
# /exports/csce/eddie, /exports/chss/eddie, /exports/cmvm/eddie, /exports/igmm/eddie or /exports/eddie/scratch
DESTINATION=""" + project_path + """

# Do the copy with rsync (avoid -p or -a options)
rsync -rm ${SOURCE} ${DESTINATION}

# Source path on DataStore. It should start with one of
# /exports/csce/datastore, /exports/chss/datastore, /exports/cmvm/datastore or /exports/igmm/datastore
SOURCE=""" + source_path_DLC_VR_PUPIL_MODEL + """

# Destination path on Eddie. It should start with one of:
# /exports/csce/eddie, /exports/chss/eddie, /exports/cmvm/eddie, /exports/igmm/eddie or /exports/eddie/scratch
DESTINATION=""" + project_path + """

# Do the copy with rsync (avoid -p or -a options)
rsync -rm ${SOURCE} ${DESTINATION}


# Source path on DataStore. It should start with one of
# /exports/csce/datastore, /exports/chss/datastore, /exports/cmvm/datastore or /exports/igmm/datastore
SOURCE=""" + source_path_DLC_VR_LICK_MODEL + """

# Destination path on Eddie. It should start with one of:
# /exports/csce/eddie, /exports/chss/eddie, /exports/cmvm/eddie, /exports/igmm/eddie or /exports/eddie/scratch
DESTINATION=""" + project_path + """

# Do the copy with rsync (avoid -p or -a options)
rsync -rm ${SOURCE} ${DESTINATION}"""


staging_text = staging_file_text(mouse, day)

f = open(code_path + "stagein_M" +mouse+"_D"+day+".sh", "w")
f.write(staging_text)
f.close()

stagein_string = "qsub stagein_M"+mouse+"_D"+day+".sh"


def make_compute_script(mouse, day):
    return """#!/bin/bash
#Grid Engine options go here, e.g.:
#$ -cwd
#$ -N M"""+mouse+"""_D"""+day+"""_PL
#$ -pe sharedmem 24 -l rl9=true,h_vmem=30G,h_rt=48:00:00
#$ -M """ + email + """ -m e
# Setup the environment modules command
source /etc/profile.d/modules.sh
# Load modules if required
# e.g. module load <name>/<version>
module load anaconda
conda activate si
date
export PYTHONPATH=""" + repo_path + """
python """ + repo_path + """run_scripts/run_pipeline_eddie.py """+mouse+""" """+day+"""
date
"""

compute_file_path = code_path + "M"+mouse+"_D"+day+".sh"
f = open(compute_file_path, "w")
f.write(make_compute_script(mouse=mouse, day=day))
f.close()
compute_string = "qsub -hold_jid M"+mouse+"_D"+day+"_DL M"+mouse+"_D"+day+".sh"

stageout_text = """
# Grid Engine options start with a #$
#$ -cwd
#$ -N M"""+mouse+"""_D"""+day+"""_UL
# Choose the staging environment
#$ -q staging

# Hard runtime limit
#$ -l h_rt=01:00:00

# Source and destination directories
#
# Source path on Eddie. It should be on the fast Eddie HPC filesystem, starting with one of:
# /exports/csce/eddie, /exports/chss/eddie, /exports/cmvm/eddie, /exports/igmm/eddie or /exports/eddie/scratch,
#
SOURCE=""" + ders_path + """
#
# Destination path on DataStore in the staging environment
# Note: these paths are only available on the staging nodes
# It should start with one of /exports/csce/datastore, /exports/chss/datastore, /exports/cmvm/datastore or /exports/igmm/datastore
#
DESTINATION=""" + output_path + """

# Perform copy with rsync
# Note: do not use -p or -a (implies -p) as this can break file ACLs at the destination
rsync -rl ${SOURCE} ${DESTINATION}"""

f = open(code_path + "stageout_M"+mouse+"_D"+day+".sh", "w")
f.write(stageout_text)
f.close()
stageout_string = "qsub -hold_jid M"+mouse+"_D"+day+"_PL stageout_M"+mouse+"_D"+day+".sh"


os.chdir(code_path)

subprocess.call( stagein_string.split() )
subprocess.call( compute_string.split() )
subprocess.call( stageout_string.split() )






