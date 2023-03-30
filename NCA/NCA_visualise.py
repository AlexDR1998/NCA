import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import subprocess
import numpy as np 
import scipy as sp
from NCA.NCA_utils import *
from tqdm import tqdm
#import cv2 
import tensorflow as tf

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
  
  def save_image_sequence_RGBA(self,trajectory,filename,zoom=4,normalise=False):
    """
      Saves a trajectory with 'frame***.png' format for each timepoint
    """
    
   
    if normalise:
      trajectory[...,:4] = trajectory[...,:4]/np.max(trajectory[...,:4])
    trajectory = sp.ndimage.zoom(trajectory,zoom=[1,1,zoom,zoom,1],order=0)
    for i in tqdm(range(trajectory.shape[0])):
      if normalise:
        plt.imsave(filename+"frame"+f"{i:03}"+".png",np.clip(trajectory[i,0,...,:4],0,1),vmin=0,vmax=1)
      else:
        plt.imsave(filename+"frame"+f"{i:03}"+".png",np.clip(trajectory[i,0,...,:4],0,1))
      #plt.savefig(filename+"frame"+f"{i:03}"+".png",bbox_inches="tight")

  def save_image_sequence_RGB(self,trajectory,filename,zoom=4,normalise=False):
    """
      Saves a trajectory with 'frame***.png' format for each timepoint
    """
    
   
    if normalise:
      trajectory[...,:3] = trajectory[...,:3]/np.max(trajectory[...,:3])
    trajectory = sp.ndimage.zoom(trajectory,zoom=[1,1,zoom,zoom,1],order=0)  
    for i in tqdm(range(trajectory.shape[0])):
      if normalise:
        plt.imsave(filename+"frame"+f"{i:03}"+".png",np.clip(trajectory[i,0,...,:3],0,1),vmin=0,vmax=1)
      else:
        plt.imsave(filename+"frame"+f"{i:03}"+".png",np.clip(trajectory[i,0,...,:3],0,1))
      #plt.savefig(filename+"frame"+f"{i:03}"+".png",bbox_inches="tight")
  
  def save_image_sequence(self,trajectory,filename,zoom=4,normalise=True):
    """
      Saves a trajectory with 'frame***.png' format for each timepoint
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





  
  def process_to_mp4(self,filename,directory):
	  cmd_str = "ffmpeg -r 15 -start_number 0 -i frame%03d.png -c:v libx264 -r 30 -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p "+filename+".mp4"
	  print(subprocess.run(["cd "+directory,cmd_str],shell=True))
 








   
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


def my_animate_label(img):
	img = np.clip(img,0,1)
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	# add another axes at the top left corner of the figure
	axtext = fig.add_axes([0.0,0.95,0.1,0.05])
	# turn the axis labels/spines/ticks off
	axtext.axis("off")

	cax1 = ax1.imshow(img[0])
	# place the text to the other axes
	time = axtext.text(0.5,0.5, str(0), ha="left", va="top")
	def animate(i):
		cax1.set_array(img[i])
		time.set_text(str(i))
		return cax1,time
	anim = animation.FuncAnimation(fig, animate, frames=img.shape[0],
                                    interval=50, blit=True)
	plt.show()
	
def my_animate_label2(img):
	img = np.clip(img,0,1)
	fig = plt.figure()
	ax1 = fig.add_subplot(1, 1, 1)
	
	cax1 = ax1.imshow(img[0])
	time = ax1.annotate(0, xy=(1, 8), xytext=(1, 8),color="white",bbox=dict(boxstyle="square"))
	
	def animate(i):
	
	    cax1.set_array(img[i])
	
	    time.set_text(str(i))
	
	    return time, cax1
	
	anim = animation.FuncAnimation(fig, animate,
	                               frames=img.shape[0], interval=50, blit=True)
	
	plt.show()