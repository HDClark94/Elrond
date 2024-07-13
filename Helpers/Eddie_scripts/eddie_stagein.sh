#!/bin/sh

# Grid Engine options can go here (always start with #$ )
# Name job and set to use current working directory
#$ -cwd
#$ -N staging

# Choose the staging queue
#$ -q staging

# Hard runtime limit
#$ -l h_rt=00:10:00

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load modules if required
module load anaconda
conda activate si

# append the python path
export PYTHONPATH="/home/hclark3/Elrond"

python /home/hclark3/Elrond/Helpers/Eddie_scripts/eddie_stagein.py
