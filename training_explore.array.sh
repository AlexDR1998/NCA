#! /bin/sh
#$ -N training_explore
#$ -cwd
#$ -l h_rt=8:00:00

#$ -pe gpu-titanx 2
#$ -l h_vmem=30G

bash training_explore.sh $SGE_TASK_ID