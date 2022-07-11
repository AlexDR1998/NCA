#! /bin/sh
#$ -N grid_search
#$ -cwd
#$ -l h_rt=8:00:00

#$ -pe gpu-titanx 2
#$ -l h_vmem=30G

bash grid_search.sh $SGE_TASK_ID $1