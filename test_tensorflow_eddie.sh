#! /bin/sh
#$ -N tensorflow_test
#$ -cwd
#$ -l h_rt=1:00:00

#$ -pe gpu 1
#$ -l h_vmem=16G

. /etc/profile.d/modules.sh

module load anaconda
source activate mphys_python
python ./test_tensorflow_eddie.py 
source deactivate