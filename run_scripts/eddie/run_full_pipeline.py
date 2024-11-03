import sys
import os
from pathlib import Path
import subprocess

python_file = sys.argv[0]
mouse = sys.argv[1]
day = sys.argv[2]
sorter_name = sys.argv[3]
project_path = sys.argv[4]

path_to_python_script_on_eddie = str(Path(python_file).parent)

if project_path[0] == '/': 
    current_directory = ''
else:
    current_directory = os.getcwd() + '/'

script_content = """#!/bin/bash
#$ -cwd -q staging -l rl9=true,h_vmem=8G,h_rt=0:29:59 -N M"""+mouse+"""_"""+day+"""_full_pipe
source /etc/profile.d/modules.sh
module load anaconda
conda activate elrond
python """ + current_directory + path_to_python_script_on_eddie + "/run_stagein_full_stageout.py " +  mouse + " " + day + " " + sorter_name + " " + project_path

overarching_file_name = current_directory +  "M" + mouse + "_" + day + "_full.sh"
if os.path.exists(overarching_file_name):
    os.remove(overarching_file_name)

stagein_file_name = current_directory +  "M" + mouse + "_" + day + "_in.sh"
if os.path.exists(stagein_file_name):
    os.remove(stagein_file_name)

pipeline_file_name = current_directory + "M" + mouse + "_" + day + "_" + sorter_name + "_pipe_full"
if os.path.exists(pipeline_file_name):
    os.remove(pipeline_file_name)

stageout_file_name = current_directory + "M" + mouse + "_" + day + "_out_" + sorter_name
if os.path.exists(stageout_file_name):
    os.remove(stageout_file_name)

print("Script file names:")
print(overarching_file_name)
print(stagein_file_name)
print(pipeline_file_name)
print(stageout_file_name)

f = open(overarching_file_name, "w")
f.write(script_content)
f.close()

compute_string = "qsub " + overarching_file_name
subprocess.run( compute_string.split() )

import time
while os.exists(stageout_file_name) is False:
    time.sleep(5)

subprocess.run( ("qsub " + stagein_file_name).split() )
subprocess.run( ("qsub " + pipeline_file_name).split() )
subprocess.run( ("qsub " + stageout_file_name).split() )
