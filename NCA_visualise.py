import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import numpy as np 
import scipy as sp
from NCA_utils import *
from tqdm import tqdm
#import cv2 
import tensorflow as tf
import seaborn as sns 
"""
	Utilities and helper functions for visualising results. Don't use this on Eddie
"""




class NCA_Visualiser(object):
  """
    Class for purposes of visualising and analysing trained NCA models
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
    
  def single_pixel(self):
    """
      Explores how simple patterns of order 1 pixel evolve
    """
    W = 100
    for M in self.NCA_models:
      x0 = np.zeros((1,W,W,M.N_CHANNELS))
      mask = np.ones((1,W,W))
      x0[:,W//2,W//2] = 1.0
      trajectory = M.run(x0,200,1,mask)
      my_animate(trajectory[:,0,...,:3])
      self.space_average(trajectory)
    return trajectory
  def space_average(self,trajectory,mode=0):
    """
      Plots spatial averages of channels over time
    """
    if mode==0:
      plt.plot(np.mean(trajectory[:,0,...,0],axis=(1,2)),label="Red",color="red")
      plt.plot(np.mean(trajectory[:,0,...,1],axis=(1,2)),label="Green",color="green")
      plt.plot(np.mean(trajectory[:,0,...,2],axis=(1,2)),label="Blue",color="blue")
      plt.plot(np.mean(trajectory[:,0,...,3],axis=(1,2)),label="Alpha",color="black")
    else:
      plt.plot(np.mean(trajectory[:,0,...,0],axis=(1,2)),label="GSC",color="red")
      plt.plot(np.mean(trajectory[:,0,...,1],axis=(1,2)),label="Brachyury T",color="green")
      plt.plot(np.mean(trajectory[:,0,...,2],axis=(1,2)),label="SOX2",color="blue")
      plt.plot(np.mean(trajectory[:,0,...,3],axis=(1,2)),label="Lamina B",color="black")
    
    plt.plot(np.mean(trajectory[:,0,...,4:],axis=(1,2)),alpha=0.2,color="purple")
    plt.plot([],label="Hidden channels",alpha=0.2,color="purple")
    plt.legend()
    plt.xlabel("Time steps")
    plt.ylabel("Average concentration")
    plt.show()
  

  def edge(self,W=20,steps=100):
    """
      Explores how initial condition of edge behaves
    """
    for M in self.NCA_models:
      x0 = np.zeros((1,W,W,M.N_CHANNELS))
      mask = np.ones((1,W,W))
      x0[:,W//2:] = 1.0
      mask[:,:W//2] = 0.0
      trajectory = M.run(x0,steps,1,mask)
      my_animate(trajectory[:,0,...,:3])
      self.space_average(trajectory)
    return trajectory
  
  def half_circular_mask(self,r,steps=100):
    """
      Runs a NCA trajectory where the circular adhesion mask has a radius scaled by r
    """
    data,mask = load_sequence_ensemble_average(rscale=r)
    data = data[:,:,::2,::2]
    mask = mask[:,::2,::2]
    W = mask.shape[1]
    mask[:,W//2:]= 0.0
    data[:,:,W//2:]=0.0
    for M in self.NCA_models:
      trajectory = M.run(data[0],steps,1,mask)
      my_animate(trajectory[:,0,...,:3])
      self.space_average(trajectory)
    return trajectory






  def circular_mask(self,r,steps=100,downsample=2):
    """
      Runs a NCA trajectory where the circular adhesion mask has a radius scaled by r
    """
    data,mask = load_sequence_ensemble_average(rscale=r)
    K=0
    data = np.pad(data[:,:,::downsample,::downsample],((0,0),(0,0),(K,K),(K,K),(0,0)))
    mask = np.pad(mask[:,::downsample,::downsample],((0,0),(K,K),(K,K)))

    
    
    for M in self.NCA_models:
      trajectory = M.run(data[0],steps,1,mask)
      my_animate(trajectory[:,0,...,:3])
      self.space_average(trajectory)
    return trajectory

  def semi_circular_mask(self,r,steps=100):
    """
      Runs a NCA trajectory with semi-circular adhesion mask and initial condition
    """


    data,mask = load_sequence_ensemble_average(rscale=r)
    W = mask.shape[1]
    mask[:,W//2:] = 0
    data[:,:,W//2:] = 0
    K=0
    data = np.pad(data[:,:,::2,::2],((0,0),(0,0),(K,K),(K,K),(0,0)))
    mask = np.pad(mask[:,::2,::2],((0,0),(K,K),(K,K)))

    for M in self.NCA_models:
      trajectory = M.run(data[0],steps,1,mask)
      print(trajectory.shape)
      my_animate(trajectory[:,0,:W//4+10,...,:3])
      self.space_average(trajectory)
    return trajectory[:,:,:W//4+10]
  
  def square_mask(self,r,steps=100):

    """
      Runs a NCA trajectory with square adhesion mask and initial condition
    """


    data,mask = load_sequence_ensemble_average(rscale=r)
    W = mask.shape[1]
    dw=2
    zs = np.zeros(mask.shape)
    zs[:,W//4-dw:3*W//4+dw,W//4-dw:3*W//4+dw] = mask[:,W//4-dw:3*W//4+dw,W//4-dw:3*W//4+dw]
    plt.imshow(zs[0])
    plt.show()
    mask = zs

    zs2 = np.zeros(data.shape)
    zs2[:,:,W//4-dw:3*W//4+dw,W//4-dw:3*W//4+dw] = data[:,:,W//4-dw:3*W//4+dw,W//4-dw:3*W//4+dw]
    plt.imshow(zs2[0,0])
    plt.show()
    data = zs2


    K=0
    data = np.pad(data[:,:,::2,::2],((0,0),(0,0),(K,K),(K,K),(0,0)))
    mask = np.pad(mask[:,::2,::2],((0,0),(K,K),(K,K)))

    for M in self.NCA_models:
      trajectory = M.run(data[0],steps,1,mask)
      print(trajectory.shape)
      my_animate(trajectory[:,0,...,:3])
      self.space_average(trajectory)
    return trajectory


  def save_image_sequence_RGB(self,trajectory,filename,normalise=True):
    """
      Saves a trajectory animation for the animate package of latex to show in presentations
    """
    
    """ # really slow
    if normalise:
      trajectory[...,:3] = trajectory[...,:3]/np.max(trajectory[...,:3])
    for i in tqdm(range(trajectory.shape[0])):
      if normalise:
        plt.imshow(np.clip(trajectory[i,0,...,:3],0,1),vmin=0,vmax=1)
      else:
        plt.imshow(np.clip(trajectory[i,0,...,:3],0,1))
      plt.savefig(filename+"frame"+f"{i:03}"+".png",bbox_inches="tight")
    """
    if normalise:
      trajectory[...,:3] = trajectory[...,:3]/np.max(trajectory[...,:3])
    for i in tqdm(range(trajectory.shape[0])):
      if normalise:
        plt.imsave(filename+"frame"+f"{i:03}"+".png",np.clip(trajectory[i,0,...,:3],0,1),vmin=0,vmax=1)
      else:
        plt.imsave(filename+"frame"+f"{i:03}"+".png",np.clip(trajectory[i,0,...,:3],0,1))
      #plt.savefig(filename+"frame"+f"{i:03}"+".png",bbox_inches="tight")
  """  
  def save_video(self,trajectory,filename):
    trajectory = np.clip(0,1,trajectory)
    trajectory[...,:3]*=trajectory[...,3:4]
    trajectory = np.floor(255*trajectory).astype(np.uint8)

    its = trajectory.shape[0]
    am=3
    frameSize = (trajectory.shape[2],trajectory.shape[3])
    print("Frame size: "+str(frameSize))
    out = cv2.VideoWriter(str(filename)+'.avi',cv2.VideoWriter_fourcc(*'FMP4'), 8, frameSize)
    for i in range(its):
        #img = np.uint8(images[i])
        out.write(trajectory[i,0,...,:3])

    out.release()
  """
  def save_image_sequence(self,trajectory,filename,zoom=4,normalise=True):
    """
      Saves a trajectory animation for the animate package of latex to show in presentations
    """
    
    """ # really slow
    if normalise:
      trajectory[...,:3] = trajectory[...,:3]/np.max(trajectory[...,:3])
    for i in tqdm(range(trajectory.shape[0])):
      if normalise:
        plt.imshow(np.clip(trajectory[i,0,...,:3],0,1),vmin=0,vmax=1)
      else:
        plt.imshow(np.clip(trajectory[i,0,...,:3],0,1))
      plt.savefig(filename+"frame"+f"{i:03}"+".png",bbox_inches="tight")
    """

    if normalise:
      trajectory = trajectory/np.max(trajectory)
    trajectory = sp.ndimage.zoom(trajectory,zoom=[1,1,zoom,zoom],order=0)
    for i in tqdm(range(trajectory.shape[0])):
      if normalise:
        plt.imsave(filename+"frame"+f"{i:03}"+".png",np.clip(trajectory[i,0],0,1),vmin=0,vmax=1)
      else:
        plt.imsave(filename+"frame"+f"{i:03}"+".png",np.clip(trajectory[i,0],0,1))
      #plt.savefig(filename+"frame"+f"{i:03}"+".png",bbox_inches="tight")
  
  def distance_from_target(self,trajectory,target):
    dist = np.zeros(trajectory.shape[0])
    print(trajectory[0].shape)
    print(target.shape)
    for t in range(trajectory.shape[0]):
      dist[t]=tf.math.reduce_euclidean_norm(trajectory[t,...,:4]-target)
    return dist




  def gradient_plot(self):
    for M in self.NCA_models:
      params = M.dense_model.trainable_variables
      print(len(params))
      
      #Consider first just the input layer after the perception convolution

      #for i in [0,2,4]:
      X = params[0].numpy()
      print(X.shape)
      N_CHANNELS = M.N_CHANNELS
      #weights_identity = X[0,0,:N_CHANNELS]
      #weights_laplacian= X[0,0,N_CHANNELS:2*N_CHANNELS]
      #weights_average = X[0,0,2*N_CHANNELS:]
      if M.KERNEL_TYPE=="ID_LAP":
        weights_identity = X[0,0,::2]
        weights_laplacian= X[0,0,1::2]
      elif M.KERNEL_TYPE=="ID_LAP_AV":
        weights_identity = X[0,0,::3]
        weights_laplacian= X[0,0,1::3]
        weights_average = X[0,0,2::3]
      #plt.plot(weights_identity)
      plt.boxplot(weights_identity.T)
      plt.xlabel("Channels")
      plt.ylabel("1st layer weights")
      plt.title("Identity kernel weights")
      plt.show()
      
      plt.boxplot(weights_laplacian.T)
      plt.xlabel("Channels")
      plt.ylabel("1st layer weights")
      plt.title("Laplacian kernel weights")
      plt.show()
      
      
      plt.boxplot(weights_average.T)
      plt.xlabel("Channels")
      plt.ylabel("1st layer weights")
      plt.title("Average kernel weights")
      plt.show()
      

  def plot_weight_matrices(self):
    # Given an NCA, return heatmaps of its weight matrices
    for M in self.NCA_models:
      weights = M.dense_model.get_weights()
      L = len(weights)-1
      fig,ax = plt.subplots(1,L)
      print(weights)
      print(L)
      for i in range(L):
        print(weights[i].shape)
        ax[i].imshow(weights[i][0,0])
      plt.show()

  def activation_heatmap_1(self):
    """
    For each kernel, plots heatmap of how each channel excites or inhibits each channel
    """
    for M in self.NCA_models:
      params = M.dense_model.trainable_variables
      print(len(params))
      N_CHANNELS = M.N_CHANNELS
      #Consider first just the input layer after the perception convolution

      #for i in [0,2,4]:
      X_0 = params[0].numpy()[0,0]
      X_1 = params[1].numpy()[0,0]
      X_2 = params[2].numpy()[0,0]
      for i in range(3):
        X_ker = X_0[i*N_CHANNELS:(i+1)*N_CHANNELS]
        print(X_ker.shape)

        print(X_1.shape)
        print(X_2.shape)
        X_ker_1 = np.einsum("ij,jk->ik",X_ker,X_1)

        #mat = np.einsum("ij,jk,kl->il",X_ker,X_1,X_2)
        def activation(x):
          return x/(1+np.exp(x))
        nonlinear_bit = activation(X_ker_1)
        mat = np.einsum("ik,kl->il",nonlinear_bit,X_2)
        plt.imshow(mat)
        plt.show()
      #N_CHANNELS = M.N_CHANNELS
      #weights_identity = X[0,0,:N_CHANNELS]
      #weights_laplacian= X[0,0,N_CHANNELS:2*N_CHANNELS]
      #weights_average = X[0,0,2*N_CHANNELS:]
  def activation_heatmap_2(self):
    """
    Plots heatmap of NCA neural network outputs for a set of test inputs

    """
    N_KERNELS = 3
    for M in self.NCA_models:
      N_CHANNELS = M.N_CHANNELS
      #M(tf.zeros([1,3,3,N_CHANNELS])) 
      heatmap = np.zeros((N_KERNELS*N_CHANNELS,N_CHANNELS))
      for n in range(N_CHANNELS*N_KERNELS):
        X = np.zeros((1,1,1,N_KERNELS*N_CHANNELS))
        X[...,n] = 1
        heatmap[n] = M.dense_model(X)[0,0,0]
        #print(M.dense_model(X)
      plt.imshow(heatmap)
      plt.show()

  #def run_activations(self,x0,steps):
    """
      Runs a NCA from x0 like NCA.run(), but outputs increments from individual kernels
    """
    #for M in self.NCA_models:


  def maximal_perturbations(self,data,iter_n,TRAIN_ITERS=1000,N_BATCHES=4):
    """
      Finds the largest perturbations to the initial condition such that subsequent states are unchanged.
      Formally find X to minimise: sum_i(d(True_i,Pred_i)) - d(Init,X)

      Parameters
      ----------
      data : float32 tensor [T,batches,size,size,4]
        The image sequence being modelled. data[0] is treated as the initial condition

      Returns
      -------
      x0 : float32 tensor [(T-1)*batches,size,size,4]
        Set of perturbed initial conditions
    """
    #--- Setup training algorithm

    lr = 2e-3
    #lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([TRAIN_ITERS//2], [lr, lr*0.1])
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(lr, TRAIN_ITERS, 0.96)
    trainer = tf.keras.optimizers.Adam(lr_sched)
    

    #--- Iterate over list of NCA models
    for M in self.NCA_models:
      
      if data.shape[1]==1:
        #If there is only 1 batch of data, repeat it along batch axis N_BATCHES times
        data = np.repeat(data,N_BATCHES,axis=1).astype("float32")
      
      #--- Pre-process initial conditions
      x0 = np.copy(data[:-1])
      x0 = x0.reshape((-1,x0.shape[2],x0.shape[3],x0.shape[4]))
      x0_true = tf.convert_to_tensor(x0, dtype=tf.float32)
      x0 = np.clip(x0 +0.01*np.random.normal(size=x0.shape),0,1)
      x0 = tf.Variable(tf.convert_to_tensor(x0,dtype=tf.float32))
      print(x0.shape)
      
      #--- Pre-process target states
      target = np.copy(data[1:])
      target = target.reshape((-1,target.shape[2],target.shape[3],target.shape[4]))
      target = tf.convert_to_tensor(target,dtype=tf.float32)
      
      #T = data.shape[0]
      z0 = tf.zeros((x0.shape[0],x0.shape[1],x0.shape[2],M.N_CHANNELS-x0.shape[3]))
      
      


      def loss(x,x0):
        """
          Loss function for training to minimise. Averages over batches, returns error per time slice (sequence)

          Parameters
          ----------
          x : float32 tensor [(T-1)*N_BATCHES,size,size,N_CHANNELS]
            Current state of NCA grids, in sequence training mode
          
          Returns
          -------
          loss : float32 tensor [T]
            Array of errors at each timestep
        """

        target_err = tf.math.reduce_euclidean_norm((x[...,:4]-target),[-2,-3,-1])
        initial_err= tf.math.reduce_euclidean_norm((x0[...,:4]-x0_true[...,:4]),[-2,-3,-1])
        initial_reg= tf.math.reduce_euclidean_norm(x0[...,:4])
        return tf.reduce_mean(tf.reshape(target_err-initial_err,(-1,N_BATCHES)),-1)+initial_reg

      def trajectory_iteration(x0):
        """
          Runs the NCA from initial condition X for steps number of steps


          Parameters
          ----------
          x0 : float32 tensor [(T-1)*N_BATCHES,size,size,4]
        """
        #x0 = tf.convert_to_tensor(x0, dtype=tf.float32)

        if M.ADHESION_MASK is not None:
          _mask = np.zeros((data[0:1].shape),dtype="float32")
          _mask[...,4]=1
          _mask = tf.convert_to_tensor(_mask, dtype=tf.float32)


        #reg_log = []
        with tf.GradientTape() as g:
          g.watch(x0)
          x0_full = tf.concat((x0,z0),axis=-1)
          #print(x0_full.shape)
          x=tf.identity(x0_full)
          for i in range(iter_n):
            x = M(x)
            if M.ADHESION_MASK is not None:
              x = _mask*M.ADHESION_MASK + (1-_mask)*x
            
            #--- Intermediate state regulariser, to penalise any pixels being outwith [0,1]
            #above_1 = tf.math.maximum(tf.reduce_max(x),1) - 1
            #below_0 = tf.math.maximum(tf.reduce_max(-x),0)
            #reg_log.append(tf.math.maximum(above_1,below_0))
            
          #print(x.shape)
          losses = loss(x,x0) 
          #reg_loss = tf.cast(tf.reduce_mean(reg_log),tf.float32)
          mean_loss = tf.reduce_mean(losses)# + REG_COEFF*reg_loss
          #print(mean_loss)
          #print(tf.reduce_sum(x0))
        grads = g.gradient(mean_loss,x0)
        #print("Gradient shape: "+str(grads.shape))
        #print(grads)
        grads = [g/(tf.norm(g)+1e-8) for g in grads]
        x0_shape = x0.shape
        #plt.imshow(x0[0])
        #plt.show()
        #plt.imshow(grads[0])
        #plt.show()
        trainer.apply_gradients(zip([grads], [x0]))
        #grads_flat = tf.reshape(grads,-1)
        #x0_flat = tf.reshape(x0,-1)
        #trainer.apply_gradients(zip(grads_flat, x0_flat))
        #x0 = tf.reshape(x0_flat,x0_shape)

        return x0
      #--- Do training loop but modifiy initial condition
      for j in tqdm(range(TRAIN_ITERS)):
        x0 = trajectory_iteration(x0)
      return x0




def my_animate(img):
  """
    Boilerplate code to produce matplotlib animation

    Parameters
    ----------
    img : float32 or int array [t,x,y,rgb]
      img must be float in range [0,1] or int in range [0,255]

  """
  img = np.clip(img,0,1)
  frames = [] # for storing the generated images
  fig = plt.figure()
  for i in range(img.shape[0]):
    frames.append([plt.imshow(img[i],vmin=0,vmax=1,animated=True)])

  ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=0)
  
  plt.show()



