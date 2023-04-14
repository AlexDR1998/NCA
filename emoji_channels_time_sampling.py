from NCA.NCA_class import *
from NCA.trainer.NCA_trainer import *
from NCA.NCA_utils import *
import sys

index=int(sys.argv[1])-1
N_BATCHES = 4
TRAIN_ITERS = 8000
LEARN_RATE = 1e-3
NCA_WEIGHT_REG = 0.001
OPTIMIZER="Nadam"

N_CHANNELS,SAMPLING = index_to_channel_sample(index)

FILENAME = "model_exploration/Nadam_euclidean_emoji_"+str(N_CHANNELS)+"_channels_"+str(SAMPLING)+"_sampling_v1"
data = load_emoji_sequence(["alien_monster.png","microbe.png","rooster_1f413.png","rooster_1f413.png"],downsample=2)
print(data)
ca = NCA(N_CHANNELS,
	     ACTIVATION="relu",
		 REGULARIZER=NCA_WEIGHT_REG,
		 LAYERS=2,
		 KERNEL_TYPE="ID_DIFF_LAP",
		 PADDING="zero")
trainer = NCA_Trainer(ca,data,N_BATCHES,model_filename=FILENAME)
trainer.data_pad_augment(2,10)
trainer.data_noise_augment(0.001)
print(ca)
trainer.train_sequence(TRAIN_ITERS,SAMPLING,OPTIMIZER=OPTIMIZER,LEARN_RATE=LEARN_RATE)