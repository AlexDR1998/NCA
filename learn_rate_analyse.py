from NCA.NCA_class import *
from NCA.trainer.NCA_train import *
from NCA.trainer.NCA_PDE_trainer import NCA_PDE_Trainer
from NCA.NCA_utils import *
from NCA.NCA_visualise import *
from NCA.NCA_analyse import *
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

def F_readif_chem_coral(X,Xdx,Xdy,Xdd,D=[0.1,0.05],f=0.06230,k=0.06258):
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

	for i in range(4):
		for j in tqdm(range(5)):
			for k in range(2):
				index = np.ravel_multi_index((i,j,k),(4,5,2))
				LOSS_FUNC,LOSS_FUNC_STRING,SAMPLING,NORM_GRADS = index_to_mitosis_parameters(index)
				try:
					#LEARN_RATE,LEARN_RATE_STRING,OPTIMIZER,TRAIN_MODE,NORM_GRADS  = index_to_learnrate_parameters(index)
					
					
					if NORM_GRADS:
						readif_filename="logs/eddie_runs/training_exploration/PDE_readif_"+OPTIMIZER+"_"+LOSS_FUNC_STRING+"_sampling_"+str(SAMPLING)+"_grad_norm/train"
						steps,losses = load_loss_log(readif_filename)	
						plt.semilogy(steps[:4000],losses[:4000],alpha=0.8,color=colours[j],linestyle="solid")
						
					#else:
						
						#readif_filename="logs/eddie_runs/training_exploration/PDE_readif_"+OPTIMIZER+"_"+LOSS_FUNC_STRING+"_sampling_"+str(SAMPLING)+"/train"
						#steps,losses = load_loss_log(readif_filename)
						#plt.semilogy(steps[:4000],losses[:4000],alpha=0.8,color=colours[j],linestyle="dashed")
						
							
				except:
					pass
			plt.plot([],[],color=colours[j],label="sampling = "+str(SAMPLING))
	
		#plt.plot([],[],color="black",linestyle="solid",label="Normalised gradients")
		#plt.plot([],[],color="black",linestyle="dashed",label="Full gradients")	
		
		plt.title("Nadam - "+LOSS_FUNC_STRING)
		plt.legend()
		plt.xlabel("Training iterations")
		plt.ylabel("Loss")
		plt.show()

def compare_NCA_PDE(LOSS_FUNC_STRING="spectral_euclidean",SAMPLING=8):
    I = 1200
    S = 64
    x0 = np.ones(shape=(1,S,S,2))
    #x0[0,24:40,24:40]=0
    
    
    x0[:,S//4:S//2,S//4:S//2] = 0
    x0[:,S//4+10:S//2-10,S//4+10:S//2-10] = 1

    #x0[:,5*S//8:6*S//8,::2] = 0
    x0[:,6*S//8:,::8] = 0
    x0[:,6*S//8:,1::8] = 0
    x0[:,6*S//8:,2::8] = 0
    x0[:,6*S//8:,3::8] = 0
	
    #x0[:,::16,6*S//8:] = 0
    #x0[:,1::16,6*S//8:] = 0
    #x0[:,2::16,6*S//8:] = 0
    #x0[:,3::16,6*S//8:] = 0
    #x0[:,4::16,6*S//8:] = 0
    #x0[:,5::16,6*S//8:] = 0
    #x0[:,6::16,6*S//8:] = 0
    #x0[:,7::16,6*S//8:] = 0
    
    #x0[:,7*S//8,::8] = 0
    #x0[:,::2,3*S//4:7*S//8]=0
    #x0[:,S//2:,S//2-2:S//2+2]=0
    
    x0[...,1] = 1-x0[...,0]
    
    plt.imshow(x0[0,:,:,0])
    plt.show()
    PDE_model = PDE_solver(F_readif_chem_coral,2,1,size=[S,S],PADDING="periodic")
    trajectory_true = PDE_model.run(iterations=I,step_size=1.0,initial_condition=x0)
   
	
    #ca = load_wrapper("training_exploration/PDE_readif_Nadam_"+LOSS_FUNC_STRING+"_sampling_"+str(SAMPLING)+"_grad_norm_v3")
    #ca = load_wrapper("training_exploration/PDE_readif_Nadam_euclidean_lr_1e-3_r_2_grad_norm")
    ca = load_wrapper("trainer_validation/Nadam_euclidean_sampling_16_coral_v2")
	#ca.PADDING="periodic"
    #ca.FIRE_RATE = 0.5
    V = NCA_Visualiser([ca])
    V.plot_weight_matrices()
    P = NCA_Perturb([ca])
    P.zero_threshold(0.05)
    V = NCA_Visualiser([ca])
    V.plot_weight_matrices()
	
    trajectory = ca.run(x0,I)
	#trajectory_2 = ca.run(x0,I)
    trajectory_true = trajectory_true[:-1]
    my_animate(trajectory_true[::10,0,...,0])
    my_animate(trajectory[::10,0,...,0])
    #my_animate(trajectory_2[::10,0,...,0])
    tra_diff = np.abs(trajectory[::20,...,0]-trajectory_true[::20,...,0])
    my_animate(tra_diff[:,0])
    #my_animate(np.abs(trajectory[::10,0,...,0]-trajectory_2[::10,0,...,0]))


    plt.plot(np.sum(np.abs(trajectory[...,:2]-trajectory_true),axis=(1,2,3,4))/(S*S))
    plt.xlabel("Timestep")
    plt.ylabel("Average (over pixels) difference")
    plt.show()

    #vis = NCA_Visualiser([ca])
    #vis.save_image_sequence(trajectory_true[::20,...,0],"figures/readif/coral/euclidean/PDE/")
    #vis.save_image_sequence(trajectory[::20,...,0],"figures/readif/coral/euclidean/NCA/")
    #vis.save_image_sequence(tra_diff,"figures/readif/coral/euclidean/Error/")
compare_NCA_PDE()
#plot_training_error_nadam_lfunc_sampling()
#run_and_save_PDE_trajectory()