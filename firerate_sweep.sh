#! /bin/sh
#$ -N firerate_sweep
#$ -cwd
#$ -l h_rt=2:00:00

#$ -pe gpu-titanx 2
#$ -l h_vmem=30G

. /etc/profile.d/modules.sh

fire_rate=$1
filename=$2
module load anaconda
source activate nca_tensorflow
python ./NCA_eddie_run.py 4000 16 7 $1 "${filename}_${fire_rate}_rate_7_channel"
source deactivate