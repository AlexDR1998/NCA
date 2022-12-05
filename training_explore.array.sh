#! /bin/sh
#$ -N training_explore_PDE
#$ -cwd
#$ -l h_rt=47:00:00

#$ -pe gpu-titanx 2
#$ -l h_vmem=32G

bash training_explore.sh $SGE_TASK_ID