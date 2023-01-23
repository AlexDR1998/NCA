from NCA_class import *
from NCA_train import *
from NCA_utils import *
from NCA_visualise import *
import numpy as np
import os 
import sys



"""
	Analyse and visualise each combination of loss function and training algorithm, on both emoji and heat equation
"""
def F_readif_2(X,Xdx,Xdy,Xdd,D=[0.1,0.05],f=0.0367,k=0.0649):
	# Reaction diffusion as described in https://www.karlsims.com/rd.html
	ch_1 = D[0]*Xdd[...,0] - X[...,1]**2*X[...,0] + f*(1-X[...,0])
	ch_2 = D[1]*Xdd[...,1] + X[...,1]**2*X[...,0] - (k+f)*X[...,1]
	return tf.stack([ch_1,ch_2],-1)


def plot_training_error_wide_sweep():
	colours = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
	for j in range(3):
		for i in tqdm(range(7)):
			for k in range(2):
				for l in range(2):
					index = np.ravel_multi_index((i,j,k,l),(7,3,2,2))
					try:
						LEARN_RATE,LEARN_RATE_STRING,OPTIMIZER,TRAIN_MODE,NORM_GRADS  = index_to_learnrate_parameters(index)
						
						if NORM_GRADS:
							readif_filename="logs/eddie_runs/training_exploration/PDE_readif_"+OPTIMIZER+"_euclidean_lr_"+LEARN_RATE_STRING+"_"+TRAIN_MODE+"_grad_norm/train"
							steps,losses = load_loss_log(readif_filename)
							if TRAIN_MODE=="full":
								plt.semilogy(steps[:4000],losses[:4000],alpha=0.8,color=colours[i],linestyle="solid")
							else:
								plt.semilogy(steps[:4000],losses[:4000],alpha=0.8,color=colours[i],linestyle="dashed")
						else:
							
							readif_filename="logs/eddie_runs/training_exploration/PDE_readif_"+OPTIMIZER+"_euclidean_lr_"+LEARN_RATE_STRING+"_"+TRAIN_MODE+"/train"
							steps,losses = load_loss_log(readif_filename)
							if TRAIN_MODE=="full":
								plt.semilogy(steps[:4000],losses[:4000],"o",alpha=0.8,color=colours[i],markeredgecolor='none')
							else:
								plt.semilogy(steps[:4000],losses[:4000],"s",alpha=0.8,color=colours[i],markeredgecolor='none')
					except:
						pass
			plt.plot([],[],color=colours[i],label=LEARN_RATE_STRING)
		
		plt.plot([],[],color="black",linestyle="solid",label="Full training, normalised gradients")
		plt.plot([],[],color="black",linestyle="dashed",label="Differential training, normalised gradients")
		plt.plot([],[],"o",color="black",label="Full training")
		plt.plot([],[],"s",color="black",label="Differential training")
		plt.title(OPTIMIZER)
		plt.legend()
		plt.xlabel("Training iterations")
		plt.ylabel("Loss")
		plt.show()

def plot_training_error_nadam():
	colours = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
	OPTIMIZER = "Nadam"

	for j in range(4):
		for i in tqdm(range(7)):
			for k in range(2):
				index = np.ravel_multi_index((i,j,k),(7,4,2))
				try:
					#LEARN_RATE,LEARN_RATE_STRING,OPTIMIZER,TRAIN_MODE,NORM_GRADS  = index_to_learnrate_parameters(index)
					LEARN_RATE,LEARN_RATE_STRING,RATIO,NORM_GRADS = index_to_Nadam_parameters(index)
					
					if NORM_GRADS:
						readif_filename="logs/eddie_runs/training_exploration/PDE_readif_"+OPTIMIZER+"_euclidean_lr_"+LEARN_RATE_STRING+"_r_"+str(RATIO)+"_grad_norm/train"
						steps,losses = load_loss_log(readif_filename)	
						plt.semilogy(steps[:4000],losses[:4000],alpha=0.8,color=colours[i],linestyle="solid")
						
					else:
						
						readif_filename="logs/eddie_runs/training_exploration/PDE_readif_"+OPTIMIZER+"_euclidean_lr_"+LEARN_RATE_STRING+"_r_"+str(RATIO)+"/train"
						steps,losses = load_loss_log(readif_filename)
						plt.semilogy(steps[:4000],losses[:4000],alpha=0.8,color=colours[i],linestyle="dashed")
						
							
				except:
					pass
			plt.plot([],[],color=colours[i],label=LEARN_RATE_STRING)
	
		plt.plot([],[],color="black",linestyle="solid",label="Normalised gradients")
		plt.plot([],[],color="black",linestyle="dashed",label="Full gradients")	
		plt.title("Nadam training behaviour - ratio = "+str(RATIO))
		plt.legend()
		plt.xlabel("Training iterations")
		plt.ylabel("Loss")
		plt.show()


def plot_training_error_nadam_lfunc_sampling():
	colours = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
	OPTIMIZER = "Nadam"

	for i in range(7):
		for j in tqdm(range(5)):
			for k in range(2):
				index = np.ravel_multi_index((i,j,k),(7,5,2))
				try:
					#LEARN_RATE,LEARN_RATE_STRING,OPTIMIZER,TRAIN_MODE,NORM_GRADS  = index_to_learnrate_parameters(index)
					LOSS_FUNC,LOSS_FUNC_STRING,SAMPLING,NORM_GRADS = index_to_mitosis_parameters(index)
					
					if NORM_GRADS:
						readif_filename="logs/eddie_runs/training_exploration/PDE_readif_"+OPTIMIZER+"_"+LOSS_FUNC_STRING+"_sampling_"+str(SAMPLING)+"_grad_norm/train"
						steps,losses = load_loss_log(readif_filename)	
						plt.semilogy(steps[:4000],losses[:4000],alpha=0.8,color=colours[j],linestyle="solid")
						
					else:
						
						readif_filename="logs/eddie_runs/training_exploration/PDE_readif_"+OPTIMIZER+"_"+LOSS_FUNC_STRING+"_sampling_"+str(SAMPLING)+"/train"
						steps,losses = load_loss_log(readif_filename)
						plt.semilogy(steps[:4000],losses[:4000],alpha=0.8,color=colours[j],linestyle="dashed")
						
							
				except:
					pass
			plt.plot([],[],color=colours[j],label="sampling = "+str(SAMPLING))
	
		plt.plot([],[],color="black",linestyle="solid",label="Normalised gradients")
		plt.plot([],[],color="black",linestyle="dashed",label="Full gradients")	
		plt.title("Nadam - "+LOSS_FUNC_STRING)
		plt.legend()
		plt.xlabel("Training iterations")
		plt.ylabel("Loss")
		plt.show()

def compare_NCA_PDE(LOSS_FUNC_STRING="euclidean",SAMPLING=8):
	I = 640
	S = 64
	x0 = np.ones(shape=(1,S,S,2))
	x0[:,S//4:S//2,S//4:S//2] = 0
	
	x0[:,3*S//4:7*S//8] = 0.5
	x0[:,:,3*S//4:7*S//8]=0

	#for i in range(10):
	#	inds = np.random.choice(S,size=4,replace=False)
	#	x0[:,min(inds[0],inds[1]):max(inds[0],inds[1]),min(inds[2],inds[3]):max(inds[2],inds[3])] = 1 - x0[:,min(inds[0],inds[1]):max(inds[0],inds[1]),min(inds[2],inds[3]):max(inds[2],inds[3])]
	x0[...,1] = 1-x0[...,0]
	#plt.imshow(x0[0,:,:,0])
	#plt.show()
	PDE_model = PDE_solver(F_readif_2,2,1,size=[S,S],PADDING="periodic")
	trajectory_true = PDE_model.run(iterations=I,step_size=1.0,initial_condition=x0)
	ca = load_wrapper("training_exploration/PDE_readif_Nadam_"+LOSS_FUNC_STRING+"_sampling_"+str(SAMPLING)+"_grad_norm")
	ca.PADDING="periodic"
	trajectory = ca.run(x0,I)
	trajectory_true = trajectory_true[:-1]
	my_animate(trajectory_true[::10,0,...,0])
	my_animate(trajectory[::10,0,...,0])
	my_animate(np.abs(trajectory[::10,0,...,0]-trajectory_true[::10,0,...,0]))


compare_NCA_PDE()
#plot_training_error_nadam_lfunc_sampling()
#run_and_save_PDE_trajectory()