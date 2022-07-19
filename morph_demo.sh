#! /bin/sh
#$ -N morph_demo
#$ -cwd
#$ -l h_rt=24:00:00

#$ -pe gpu-titanx 2
#$ -l h_vmem=30G


. /etc/profile.d/modules.sh

module load anaconda
source activate nca_tensorflow

python ./emoji_morph.py $1
source deactivate