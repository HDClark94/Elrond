#!/bin/bash
Grid Engine options go here, e.g.:
#$ -cwd
#$ -pe sharedmem 8 -l rl9=true,h_vmem=30G,h_rt=32:00:00
#$ -M chalcrow@ed.ac.uk -m e

#load the environment modules
. /etc/profile.d/modules.sh

# Load modules if required
module load anaconda
conda activate si

# append the python path
export PYTHONPATH="/home/hclark3/Elrond"

# Run the program
python /home/hclark3/Elrond/Helpers/Eddie_scripts/eddie_pipeline.py