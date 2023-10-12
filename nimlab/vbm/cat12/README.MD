# CAT 12 VBM Pipeline

This bad-boy is going to take you from T1 Nifti Files to Smoothed, Warped, Modulated, Segmented, and Total Intracranial Volume Corrected nifti files. 

1) Set the working directory to be the base directory that has the: /vbm folder in it.
2) This working directory should be 'nimlab' if you cloned from my repo.

**Copy-Paste the Below Command to Run on Your Favourite Cluster**
___
> bsub -q big-multi -n 8 -R 'rusage[mem=64000] span[ptile=8]' -M 64000 -o ~/terminal_outputs/cat12_output.txt -J 'CAT12_Pipeline' /PHShome/cu135/github_repository/Research/nimlab/vbm/cat12/scripts/docker_run.sh


If you need singularity:
> conda activate nimlab
> bsub -q big-multi -n 8 -R 'rusage[mem=64000] span[ptile=8]' -M 64000 -o ~/terminal_outputs/cat12_output.txt -J 'CAT12_Pipeline' /PHShome/cu135/github_repository/Research/nimlab/vbm/cat12/scripts/singularity_run.sh
