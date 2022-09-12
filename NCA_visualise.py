import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import numpy as np 
from NCA_utils import *
from tqdm import tqdm
import cv2 
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
  def space_average(self,trajectory):
    """
      Plots spatial averages of channels over time
    """
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


  def save_image_sequence(self,trajectory,filename):
    """
      Saves a trajectory animation for the animate package of latex to show in presentations
    """
    for i in tqdm(range(trajectory.shape[0])):
      plt.imshow(np.clip(trajectory[i,0,...,:3],0,1))
      plt.savefig(filename+"frame"+f"{i:03}"+".png",bbox_inches="tight")
  
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

  def distance_from_target(self,trajectory,target):
    dist = np.zeros(trajectory.shape[0])
    print(trajectory[0].shape)
    print(target.shape)
    for t in range(trajectory.shape[0]):
      dist[t]=tf.math.reduce_euclidean_norm(trajectory[t,...,:4]-target)
    return dist

def my_animate(img):
  """
    Boilerplate code to produce matplotlib animation

    Parameters
    ----------
    img : float32 or int array [t,x,y,rgb]
      img must be float in range [0,1] or int in range [0,255]

  """
  frames = [] # for storing the generated images
  fig = plt.figure()
  for i in range(img.shape[0]):
    frames.append([plt.imshow(img[i],animated=True)])

  ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=0)
  
  plt.show()



