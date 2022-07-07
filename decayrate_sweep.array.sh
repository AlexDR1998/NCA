#! /bin/sh
#$ -N decayrate_sweep
#$ -cwd
#$ -l h_rt=8:00:00

#$ -pe gpu-titanx 2
#$ -l h_vmem=30G

bash decayrate_sweep.sh $SGE_TASK_ID $1