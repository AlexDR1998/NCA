#! /bin/sh
#$ -N grid_search
#$ -cwd
#$ -l h_rt=4:00:00

#$ -pe gpu-titanx 2
#$ -l h_vmem=30G

. /etc/profile.d/modules.sh


index=$1
filename=$2

module load anaconda
source activate nca_tensorflow

params=$(python ./parameter_index.py $index)
FIRE_RATE=$(echo $params| cut -d' ' -f 1)
#DECAY_RATE=$(echo $params| cut -d' ' -f 2)
N_CHANNELS=$(echo $params| cut -d' ' -f 2)

python ./NCA_eddie_run.py 4000 16 $N_CHANNELS $FIRE_RATE 20 "${filename}_2layer_ch${N_CHANNELS}_fr${FIRE_RATE}"
source deactivate