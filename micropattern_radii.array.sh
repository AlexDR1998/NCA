#! /bin/sh
#$ -N micropattern_radii_sizes
#$ -P scs_schumacher-group 
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 4 -l h_vmem=64G

bash micropattern_radii.sh $SGE_TASK_ID