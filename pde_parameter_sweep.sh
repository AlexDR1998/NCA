#! /bin/sh
#$ -N pde_parameters
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 4 -l h_vmem=80G


. /etc/profile.d/modules.sh


export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
module load anaconda
source activate nca_tensorflow
python ./pde_parameter_explore.py $1
source deactivate