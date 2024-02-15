#! /bin/sh
#$ -N pde_noise
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=24:00:00

#$ -q gpu -l gpu=1 -pe sharedmem 4 -l h_vmem=80G


. /etc/profile.d/modules.sh


export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
module load anaconda
source activate tf_a100
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib


python ./pde_noise_explore.py $1
source deactivate