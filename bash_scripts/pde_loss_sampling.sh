#! /bin/sh
#$ -N pde_loss_sampling
#$ -P scs_schumacher-group 
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=16:00:00

#$ -pe gpu-a100 2
#$ -l h_vmem=80G


. /etc/profile.d/modules.sh

module load anaconda
source activate tf_a100
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

python ./pde_loss_sampling.py $1
source deactivate