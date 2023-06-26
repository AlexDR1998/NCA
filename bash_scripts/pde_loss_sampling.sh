#! /bin/sh
#$ -N pde_loss_sampling
#$ -P scs_schumacher-group 
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00

#$ -pe gpu-titanx 1
#$ -l h_vmem=64G


. /etc/profile.d/modules.sh

module load anaconda
source activate nca_tensorflow

python ./pde_loss_sampling.py $1
source deactivate