#! /bin/sh
#$ -N training_explore_PDE
#$ -cwd
#$ -l h_rt=47:00:00

#$ -pe gpu-titanx 4
#$ -l h_vmem=32G


. /etc/profile.d/modules.sh

module load anaconda
source activate nca_tensorflow

python ./training_exploration.py $1
source deactivate