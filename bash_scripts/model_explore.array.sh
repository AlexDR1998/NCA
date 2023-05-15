#! /bin/sh
#$ -N model_symmetries
#$ -P scs_schumacher-group 
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=47:00:00

#$ -pe gpu-titanx 1
#$ -l h_vmem=64G

bash model_explore.sh $SGE_TASK_ID