#! /bin/sh
#$ -N NCA_test
#$ -cwd
#$ -l h_rt=8:00:00

#$ -pe gpu-titanx 4
#$ -l h_vmem=20G

. /etc/profile.d/modules.sh

channels=$1
filename=$2
module load anaconda
source activate nca_tensorflow
python ./NCA_eddie_run.py 4000 8 $1 "${filename}_${channels}_channel"
source deactivate