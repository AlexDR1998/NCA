#! /bin/sh
#$ -N NCA_train
#$ -cwd
#$ -l h_rt=8:00:00

#$ -l h_vmem=16G

bash NCA_eddie_run.sh $SGE_TASK_ID $1