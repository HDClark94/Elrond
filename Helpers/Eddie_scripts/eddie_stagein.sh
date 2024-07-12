#!/bin/sh

# Grid Engine options can go here (always start with #$ )
# Name job and set to use current working directory
#$ -cwd
#$ -N staging

# Choose the staging queue
#$ -q staging

# Hard runtime limit
#$ -l h_rt=00:10:00

# Load modules if required
# e.g. module load <name>/<version>
module load anaconda/2024.02
conda activate /exports/eddie/scratch/hclark3/anaconda/envs/si

# append the python path
export PYTHONPATH="/home/hclark3/Elrond"

python /home/hclark3/Elrond/Helpers/Eddie_scripts/eddie_stagein.py
