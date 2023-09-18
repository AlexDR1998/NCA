#! /bin/sh
#$ -N pde_loss_sampling
#$ -P scs_schumacher-group 
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=1:00:00

#$ -pe gpu-a100 1
#$ -l h_vmem=80G


. /etc/profile.d/modules.sh

module load anaconda
source activate jax_gpu
python ./jax_eddie_test.py
source deactivate