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
#$ -cwd -q staging -l rl9=true,h_vmem=8G,h_rt=0:29:59 -N full_pipe
source /etc/profile.d/modules.sh
module load anaconda
conda activate elrond
python """ + current_directory + path_to_python_script_on_eddie + "/run_stagein_full_stageout.py " +  mouse + " " + day + " " + sorter_name + " " + project_path

script_file_path = "M" + mouse + "_" + day + "_full.sh"

f = open(script_file_path, "w")
f.write(script_content)
f.close()

compute_string = "qsub " + script_content
subprocess.run( compute_string.split() )
