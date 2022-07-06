#! /bin/sh
#$ -N NCA_train
#$ -cwd
#$ -l h_rt=8:00:00

#$ -pe gpu-titanx 2
#$ -l h_vmem=30G

bash firerate_sweep.sh $SGE_TASK_ID $1