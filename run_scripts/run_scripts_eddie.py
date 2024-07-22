import os
from os.path import expanduser

import sys

mouse= sys.argv[1]
day = sys.argv[2]
print("mouse: ", mouse)
print("day: ", day)

home_path = expanduser("~")
chalcrow_path = home_path + "/../../../exports/eddie/scratch/chalcrow/"
project_path = chalcrow_path + "harry_project/"
data_path = project_path + "data/M" + mouse + "_D" + day + "/"
code_path = project_path + "code/M" + mouse + "/D" + day + "/"
ders_path = project_path + "derivatives/M" + mouse + "/D" + day + "/"

def staging_file_text(mouse, day):

    mouse_day_dir = "M"+mouse + "_" +"D"+ day

    return """#!/bin/sh

# Grid Engine options can go here (always start with #$ )
# Name job and set to use current working directory
#$ -cwd
#$ -N stg_"""+mouse_day_dir+"""

# Choose the staging queue
#$ -q staging

# Hard runtime limit
#$ -l h_rt=01:00:00

mkdir -p /exports/eddie/scratch/chalcrow/harry_project/data/M"""+mouse+"""_D"""+day+"""/

# Source path on DataStore. It should start with one of
# /exports/csce/datastore, /exports/chss/datastore, /exports/cmvm/datastore or /exports/igmm/datastore
SOURCE=/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Harry/Cohort11_april2024/

# Destination path on Eddie. It should start with one of:
# /exports/csce/eddie, /exports/chss/eddie, /exports/cmvm/eddie, /exports/igmm/eddie or /exports/eddie/scratch
DESTINATION=""" + data_path + """

# Do the copy with rsync (avoid -p or -a options)
rsync -rm --exclude='*/*/processed/' --exclude='*.avi' --include='of/' --include='of/"""+mouse_day_dir+"""*' --include='of/"""+mouse_day_dir+"""*/****' --include='vr/' --include='vr/"""+mouse_day_dir+"""*' --include='vr/"""+mouse_day_dir+"""*/****' --exclude="*" ${SOURCE} ${DESTINATION}
    """

staging_text = staging_file_text(mouse, day)

f = open(code_path + "stagein_M" +mouse+"_D"+day+".sh", "w")
f.write(staging_text)
f.close()


stagein_string = "qsub stagein_M"+mouse+"_D"+day+".sh"


def make_compute_script(mouse, day):

    mouse_day = mouse + "_" + day

    return """#!/bin/bash
#Grid Engine options go here, e.g.:
#$ -cwd
#$ -N M"""+mouse+"""_D"""+day+"""_compute
#$ -pe sharedmem 24 -l rl9=true,h_vmem=30G,h_rt=48:00:00
#$ -M chalcrow@ed.ac.uk -m e

# Setup the environment modules command
source /etc/profile.d/modules.sh

# Load modules if required
# e.g. module load <name>/<version>
module load anaconda/2024.02
conda activate si

date
cd /exports/eddie/scratch/chalcrow/harry_project/GIT/Elrond/
python run_pipeline_chris_eddie.py """+mouse+""" """+day+"""
date

# Job payload - run scripts, commands etc.
pwd       # print the working directory
free      # print the amount of available memory
hostname  # print the host the job is running on
    """

compute_file_path = code_path + "M"+mouse+"_D"+day+".sh"

f = open(compute_file_path, "w")
f.write(make_compute_script(mouse=mouse, day=day))
f.close()

compute_string = "qsub -hold_jid stg_M"+mouse+"_D"+day+ " M"+mouse+"_D"+day+".sh"

stageout_text = """

# Grid Engine options start with a #$
#$ -cwd
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
DESTINATION=/exports/cmvm/datastore/sbms/groups/CDBS_SIDB_storage/NolanLab/ActiveProjects/Chris/Cohort11_april2024/derivatives/M"""+mouse+"""/D"""+day+"""/

# Perform copy with rsync
# Note: do not use -p or -a (implies -p) as this can break file ACLs at the destination
rsync -rl ${SOURCE} ${DESTINATION}"""

f = open(code_path + "stageout_M"+mouse+"_D"+day+".sh", "w")
f.write(stageout_text)
f.close()

stageout_string = "qsub -hold_jid M"+mouse+"_D"+day+"_compute stageout_M"+mouse+"_D"+day+".sh"
print("qsub -hold_jid M"+mouse+"_D"+day+"_compute stageout_M"+mouse+"_D"+day+".sh")

import subprocess

os.chdir(code_path)

subprocess.call( stagein_string.split() )
subprocess.call( compute_string.split() )
subprocess.call( stageout_string.split() )













