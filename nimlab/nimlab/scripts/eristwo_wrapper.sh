# Wrapper for job submissions to eristwo
# module load MATLAB/2020b
# module load singularity/3.7.0
# source activate nimlab
# export MATLABPATH=/data/nimlab/software/Lesion_Quantification_Toolkit/Functions:$MATLABPATH
export PATH=/usr/share/lsf/9.1/linux2.6-glibc2.3-x86_64/bin/:$PATH:$HOME/PALM
export LSF_ENVDIR=/usr/share/lsf/conf
export LSF_SERVERDIR=/usr/share/lsf/9.1/linux2.6-glibc2.3-x86_64/etc
# export TMPDIR=$HOME/scratch
$@
