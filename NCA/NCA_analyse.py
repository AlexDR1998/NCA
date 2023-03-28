import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import numpy as np 
from NCA_utils import *
from tqdm import tqdm
import tensorflow as tf 
from NCA_analyse_utils import *
"""
	Class for interpreting trained NCA models.
"""



class NCA_Perturb(object):
  """
    Class to explore NCA models by peturbing them in various ways
  """
  def __init__(self,NCA_models=None):
    """
      Init method

      Parameters
      ----------
      NCA_model : list of NCA class
        Trained NCA models
    """
    self.NCA_models = NCA_models
 
  def zero_threshold(self,threshold):
    """
      Takes a trained model and sets all weights below threshold to 0. If a high threshold value
      still results in qualitatively similar behaviour, the model has some redundancy and
      robustness. A simpler 'pruned' model that behaves similarly is easier to interpret.
    """
    for M in self.NCA_models:
      weights = M.dense_model.get_weights()
      for i in range(len(weights)):
        weights[i] = np.where(np.abs(weights[i])>threshold,weights[i],0)
      M.dense_model.set_weights(weights)
      #plot_weight_matrices(M)
  
  def update_gradients(self,x0,steps,L):
    """
      Takes a trained model and initial condition, and returns matrices of network gradients
      with respect to change from one timestep to the next.
    """

    for M in self.NCA_models:
      x0 = M.run(x0,2)[0]
      x0 = tf.convert_to_tensor(x0)
      weights_0 = M.dense_model.weights[L][0,0]
      grads = np.zeros((steps,weights_0.shape[0],weights_0.shape[1],M.N_CHANNELS))
      for t in tqdm(range(steps)):
        for i in range(M.N_CHANNELS):
          with tf.GradientTape() as g:
            x1 = M(x0)
            diff = (x0-x1)[0,...,i]
            #print(diff.shape)
            grad = g.gradient(diff,M.dense_model.weights)
            grads[t,...,i] = grad[L]
          x0 = tf.identity(x1)
      
      return grads
        #print(grad[0].shape)
        #plt.imshow(grad[0][0,0])
        #plt.show()
      #grads = grads / np.max(grads[...,:4])
      #my_animate(grads[...,:4])



