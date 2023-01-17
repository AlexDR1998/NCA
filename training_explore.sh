#! /bin/sh
#$ -N rdiff_nadam_sweep
#$ -P scs_schumacher-group 
#$ -M s1605376@ed.ac.uk
#$ -cwd
#$ -l h_rt=47:00:00

#$ -pe gpu-titanx 1
#$ -l h_vmem=64G


. /etc/profile.d/modules.sh

module load anaconda
source activate nca_tensorflow

python ./learn_rate_sweep.py $1
source deactivate