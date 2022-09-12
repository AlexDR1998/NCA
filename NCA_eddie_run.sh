#! /bin/sh
#$ -N NCA_train
#$ -cwd
#$ -l h_rt=2:00:00

#$ -pe gpu-titanx 2
#$ -l h_vmem=30G

. /etc/profile.d/modules.sh

channels=$1
filename=$2
module load anaconda
source activate nca_tensorflow
python ./NCA_eddie_run.py 4000 16 $1 20 "${filename}_${channels}_channel"
source deactivate