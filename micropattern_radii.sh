#! /bin/sh
#$ -N micropattern_radii_random
#$ -P scs_schumacher-group 
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 4 -l h_vmem=64G



. /etc/profile.d/modules.sh
export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu

module load anaconda
source activate jax_gpu
python ./micropattern_radii_eddie.py $1
source deactivate