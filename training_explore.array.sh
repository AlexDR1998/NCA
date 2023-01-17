#! /bin/sh
#$ -N rdiff_nadam_sweep
#$ -P scs_schumacher-group 
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=47:00:00

#$ -pe gpu-titanx 1
#$ -l h_vmem=64G

bash training_explore.sh $SGE_TASK_ID