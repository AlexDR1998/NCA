import os
import io
import skimage.measure as measure
import skimage
import base64
import zipfile
import json
import requests
import numpy as np
import scipy as sp
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob

#impath = "../Data/time_course_disk_4channel/lowres_zsum/"
impath = "../Data/time_course_disk_4channel/lowres_zmax/"

def load_sequence_A(name):
  """
    Loads a single batch of timesteps of experimental data

    Parameters
    ----------
    name : string
      the name of the file

    Returns
    -------
    data : float32 array [T,1,size,size,4]
      timesteps (T) of RGBA images. Dummy index of 1 for number of batches
  """


  I_0h = skimage.io.imread(impath+name+"_0h.ome.tiff")#[::2,::2]
  I_24h = skimage.io.imread(impath+name+"_24h.ome.tiff")#[::2,::2]
  I_36h = skimage.io.imread(impath+name+"_36h.ome.tiff")#[::2,::2]
  I_48h = skimage.io.imread(impath+name+"_48h.ome.tiff")#[::2,::2]
  #I_60h = skimage.io.imread(impath+name+"_60h.ome.tiff")#[::2,::2]
  I_0h = I_0h[np.newaxis]/np.max(I_0h,axis=(0,1))
  I_24h = I_24h[np.newaxis]/np.max(I_24h,axis=(0,1))
  I_36h = I_36h[np.newaxis]/np.max(I_36h,axis=(0,1))
  I_48h = I_48h[np.newaxis]/np.max(I_48h,axis=(0,1))
  #I_60h = I_60h[np.newaxis]/np.max(I_60h,axis=(0,1))
  data = np.stack((I_0h,I_24h,I_36h,I_48h))
  return data


def load_sequence_B(name):
  """
    Loads a single batch of timesteps of experimental data. Same as load_sequence_A,
    except the B dataset has different (less) time slices

    Parameters
    ----------
    name : string
      the name of the file

    Returns
    -------
    data : float32 array [T,1,size,size,4]
      timesteps (T) of RGBA images. Dummy index of 1 for number of batches
  """


  I_24h = skimage.io.imread(impath+name+"_24h.ome.tiff")#[::2,::2]
  I_36h = skimage.io.imread(impath+name+"_36h.ome.tiff")#[::2,::2]
  I_48h = skimage.io.imread(impath+name+"_48h.ome.tiff")#[::2,::2]
  
  
  I_24h = I_24h[np.newaxis]/np.max(I_24h)
  I_36h = I_36h[np.newaxis]/np.max(I_36h)
  I_48h = I_48h[np.newaxis]/np.max(I_48h)
  data = np.stack((I_24h,I_36h,I_48h))
  return data

def load_sequence_batch(N_BATCHES):

  """
    Loads a randomly selected batch of image sequences

    Parameters
    ----------
    N_BATCHES : int
      How many batches of image sequences to load

    Returns
    -------
    data_batches : flaot32 array [T,N_BATCHES,size,size,4]
      Batches of image sequences, stacked along dimension 1
  """

  names_A = ["A1_F1","A1_F2","A1_F3","A1_F4","A1_F5",
           "A1_F6","A1_F7","A1_F8","A1_F9","A1_F10",
           "A1_F11","A1_F12","A1_F13","A1_F14","A1_F15"]
  names_selected = np.random.choice(names_A,N_BATCHES,replace=False)
  data_batches = load_sequence_A(names_selected[0])
  for i in range(1,N_BATCHES):
    data_batches = np.concatenate((data_batches,load_sequence_A(names_selected[i])),axis=1)
  return data_batches

def load_sequence_ensemble_average():
  """
    Loads all image sequences and averages across them,
    to create image sequence of ensemble averages

    Returns
    -------
    data : float32 array [T,size,size,4]
  """

  names_A = ["A1_F1","A1_F2","A1_F3","A1_F4","A1_F5",
           "A1_F6","A1_F7","A1_F8","A1_F9","A1_F10",
           "A1_F11","A1_F12","A1_F13","A1_F14","A1_F15"]
  data_batches = load_sequence_A(names_A[0])
  for i in range(1,len(names_A)):
    data_batches = np.concatenate((data_batches,load_sequence_A(names_A[i])),axis=1)
  data = np.mean(data_batches,axis=1,keepdims=True)
  return data


def adhesion_mask(data):
  """
    Given data output from load_sequence_*, returns a binary mask representing the circle where cells can adhere
    
    Parameters
    ----------
    data : float32 array [T,1,size,size,4]
      timesteps (T) of RGBA images. Dummy index of 1 for number of batches

    Returns
    -------
    mask : boolean array [1,size,size]
      Array with circle of 1/0 indicating likely presence/lack of adhesive surface in micropattern
  """

  thresh = np.mean(data[0,0],axis=-1)
  thresh = sp.ndimage.gaussian_filter(thresh,5)
  
  
  k = thresh>np.mean(thresh)
  
  regions = measure.regionprops(measure.label(k))
  cell_culture = regions[0]

  y0, x0 = cell_culture.centroid
  r = cell_culture.major_axis_length / 2.

  def cost(params):
      x0, y0, r = params
      coords = skimage.draw.disk((y0, x0), r, shape=k.shape)
      template = np.zeros_like(k)
      template[coords] = 1
      return -np.sum(template == k)

  x0, y0, r = sp.optimize.fmin(cost, (x0, y0, r))
  mask = np.zeros(k.shape,dtype="float32")
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      mask[i,j] = (i-x0)**2+(j-y0)**2<r**2
  print(mask.shape)
  return mask[np.newaxis]



def adhesion_mask_batch(data):
  """
    Applies ashesion_mask but to a batch of different initial conditions
  
    Parameters
    ----------
    data : float32 array [T,N_BATCHES,size,size,4]
      Batch of N_BATCHES image sequences

    Returns
    -------
    masks : boolean array [N_BATCHES,size,size]
      Batch of adhesion masks corresponding to each image sequence

  """



  N_BATCHES = data.shape[1]
  mask0 = adhesion_mask(data[:,0:1])
  print(mask0.shape)
  masks = np.repeat(mask0,N_BATCHES,axis=0)#np.zeros((N_BATCHES,mask0.shape[0],mask0.shape[1]))
  print(masks.shape)
  for i in range(1,N_BATCHES):
    masks[i] = adhesion_mask(data[:,i:i+1])
  return masks









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


