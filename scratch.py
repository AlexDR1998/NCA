from NCA_class import *
from NCA_utils import *
import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'







def main():

	T=0
	N_CHANNELS=16 # Must be greater than 4
	N_BATCHES=4
	

	#a = load_sequence_A("A1_F11")
	#a = load_sequence_batch(N_BATCHES)[:,:,::2,::2]
	a = load_sequence_ensemble_average()[:,:,::2,::2]
	#for i in range(5):
	#	plt.imshow(a[T,i,...,3])
	#	plt.show()
	
	print(a.shape)
	mask = adhesion_mask(a)
	print(mask.shape)
	
	#for i in range(N_BATCHES):
	#	plt.imshow(a[T,i,...,:3])
	#	plt.show()
	#	plt.imshow(mask[i])
	#	plt.show()
	
	print("Average images:")
	for i in range(4):
		plt.imshow(a[i,0,:,:,:3])
		plt.show()

	ca = NCA(N_CHANNELS,ADHESION_MASK=mask)


	#print(np.max(target[...,3]))
	#train(ca,a[3],N_BATCHES,100,a[0],50,"visualisation_test_1")
	train_sequence(ca,a,N_BATCHES,100,48,"sequence_test1")
	print("Training complete")
	#t1 = datetime.datetime.now()

	
	#ca.save_wrapper("save_test_2")
	#t2 = datetime.datetime.now()
	#print(t2-t1)
	#ca2 = tf.keras.models.load_model("save_test",custom_objects={"NCA":NCA})
	ca2 = load_wrapper("sequence_test1")
	#print(a[0].shape)

	#grids = run(ca,1000,1,100)[:,0]
	grids = ca.run(a[0],100,1,mask)[:,0]
	print(np.max(grids))
	print(np.min(grids))
	my_animate(grids[...,:4])

	grids2 = ca2.run(a[0],100,1,mask)[:,0]
	my_animate(grids2[...,:4])


	grids[...,4:] = (1+np.tanh(grids[...,4:]))/2.0
	#my_animate(np.abs(grids[...,:4]-target))
	my_animate(grids[...,4:8])
	my_animate(grids[...,8:12])
	my_animate(grids[...,12:])
	plt.imshow(grids[50,...,:4])
	plt.show()
	#my_animate((grids[...,:4]+1)/2.0)

	#error = np.sqrt(np.sum((target-grids[50,:,:,:4])**2))
	#print(error)
	#grids = (grids+np.abs(np.min(grids)))/(2*np.max(grids))
	
	# Visualise different subsets of channels 
	
	#my_animate(grids[...,3:6])
	#my_animate(grids[...,6:9])
	
main()