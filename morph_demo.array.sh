#! /bin/sh
#$ -N morph_demo
#$ -cwd
#$ -l h_rt=8:00:00

#$ -pe gpu-titanx 2
#$ -l h_vmem=30G

bash morph_demo.sh $SGE_TASK_ID