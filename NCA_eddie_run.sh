#! /bin/sh
#$ -N NCA_test
#$ -cwd
#$ -l h_rt=8:00:00

#$ -pe gpu 1
#$ -l h_vmem=16G

. /etc/profile.d/modules.sh

channels=$1
filename=$2
module load anaconda
source activate mphys_python
python ./NCA_eddie_run.py 4000 8 $1 "${channels} channel ${filename}"
source deactivate