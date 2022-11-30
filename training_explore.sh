#! /bin/sh
#$ -N training_explore
#$ -cwd
#$ -l h_rt=24:00:00

#$ -pe gpu-titanx 2
#$ -l h_vmem=32G


. /etc/profile.d/modules.sh

module load anaconda
source activate nca_tensorflow

python ./training_exploration.py $1
source deactivate