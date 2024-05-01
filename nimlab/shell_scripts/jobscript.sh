#!/bin/bash
#SBATCH --partition=nimlab,normal # partition (queue)
#SBATCH -c 1 # number of cores
#SBATCH --mem 1000 # memory pool for all cores
#SBATCH -t 0-1:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-user=choward12@bwh.harvard.edu
#SBATCH --mail-type=END,FAIL
# Valid type values are NONE, BEGIN, END, FAIL, REQUEUE, ALL (Please avoid: Equivalent to BEGIN, END, FAIL, INVALID_DEPEND, REQUEUE, and STAGE_OUT), INVALID_DEPEND (dependency never satisfied), STAGE_OUT (burst buffer stage out and teardown completed), TIME_LIMIT, TIME_LIMIT_90 (reached 90 percent of time limit), TIME_LIMIT_80 (reached 80 percent of time limit), TIME_LIMIT_50 (reached 50 percent of time limit) and ARRAY_TASKS (Please also avoid: Send emails for each array task).
# ROSETTA: https://slurm.schedmd.com/rosetta.pdf
for i in {1..1}; do
python ~/../../data/nimlab/software/software_env/python_modules/nimlab/nimlab/scripts/connectome_quick.py -o ~/connectome_outputs/ad_vta_redo -r ~/file_paths/paths_to_ad_redo.csv -cs ~/../../data/nimlab/connectome_npy/yeo1000_dil -c matrix -w 4
done