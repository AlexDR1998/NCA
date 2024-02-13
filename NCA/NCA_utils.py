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
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.framework import tensor_util
import tensorflow as tf

"""
  Utilities and helper functions to handle loading and preprocessing of data
"""

#impath = "../Data/time_course_disk_4channel/lowres_zsum/"
impath = "../Data/time_course_disk_4channel/lowres_zmax/"
impath_emojis = "../Data/Emojis/"




def load_loss_log(summary_dir):
  """
    Returns the loss logged in tensorboard as an array

    Parameters
    ----------
    summary_dir : string
      The directory where the tensorboard log is stored

    Returns
    -------
      steps : array ints
        timesteps
      losses : array float32
        losses at timesteps
  """
  def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
      yield event_pb2.Event.FromString(r)
  steps = []
  losses =[]
  
  
  
  for filename in os.listdir(summary_dir):
    if filename!="plugins":
      path = os.path.join(summary_dir, filename)
      for event in my_summary_iterator(path):
        for value in event.summary.value:
          if value.tag=="Mean Loss":
            t = tensor_util.MakeNdarray(value.tensor)
            steps.append(event.step)
            losses.append(t)
  return steps,losses

def load_trajectory_log(summary_dir):
  """
    Returns the NCA states at target times

    Parameters
    ----------
    summary_dir : string
      The directory where the tensorboard log is stored

    Returns
    -------
      steps : array ints
        timesteps
      losses : array float32
        losses at timesteps
  """
  def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
      yield event_pb2.Event.FromString(r)
  steps = []
  trajectory =[]
  
  
  
  for filename in os.listdir(summary_dir):
    if filename!="plugins":
      path = os.path.join(summary_dir, filename)
      for event in my_summary_iterator(path):
        for value in event.summary.value:
          if value.tag=="Trained NCA dynamics RGBA":
            t = tensor_util.MakeNdarray(value.tensor)
            steps.append(event.step)
            trajectory.append(t)
  return steps,trajectory

def load_emoji_sequence(filename_sequence,impath_emojis="../Data/Emojis/",downsample=2,crop_square=False):
  """
    Loads a sequence of images in impath_emojis
    Parameters
    ----------
    filename_sequence : list of strings
      List of names of files to load
    downsample : int
      How much to downsample the resolution - highres takes ages

    Returns
    -------
    images : float32 array [T,1,size,size,C]
      Timesteps of T RGB/RGBA images. Dummy index of 1 for number of batches
  """
  images = []
  for filename in filename_sequence:
    im = skimage.io.imread(impath_emojis+filename)[::downsample,::downsample]
    if crop_square:
      s= min(im.shape[0],im.shape[1])
      im = im[:s,:s]
    im = im[np.newaxis] / 255.0
    images.append(im)
  return np.array(images)

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


  I_0h = skimage.io.imread(impath+"0h/"+name+"_0h.ome.tiff")#[::2,::2]
  I_24h = skimage.io.imread(impath+"24h/A/"+name+"_24h.ome.tiff")#[::2,::2]
  I_36h = skimage.io.imread(impath+"36h/A/"+name+"_36h.ome.tiff")#[::2,::2]
  I_48h = skimage.io.imread(impath+"48h/A/"+name+"_48h.ome.tiff")#[::2,::2]
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

def load_sequence_ensemble_average(masked=True,rscale=1.0):
  """
    Loads all image sequences and averages across them,
    to create image sequence of ensemble averages

    Parameters
    ----------
    masked : boolean
      controls whether to apply the adhesion mask to the data

    rscale : float32
      scales how much bigger or smaller the radius of the mask is

    Returns
    -------
    data : float32 array [T,size,size,4]

    mask : boolean array [size,size]
  """

  """
  names_A = ["A1_F1","A1_F2","A1_F3","A1_F4","A1_F5",
           "A1_F6","A1_F7","A1_F8","A1_F9","A1_F10",
           "A1_F11","A1_F12","A1_F13","A1_F14","A1_F15"]
  data_batches = load_sequence_A(names_A[0])
  for i in range(1,len(names_A)):
    data_batches = np.concatenate((data_batches,load_sequence_A(names_A[i])),axis=1)
  data = np.mean(data_batches,axis=1,keepdims=True)
  """

  I_0h = skimage.io.imread(impath+"ensemble_averages/A/AVG_0h.tif")
  I_24h = skimage.io.imread(impath+"ensemble_averages/A/AVG_24h.tif")
  I_36h = skimage.io.imread(impath+"ensemble_averages/A/AVG_36h.tif")
  I_48h = skimage.io.imread(impath+"ensemble_averages/A/AVG_48h.tif")

  I_0h = I_0h[np.newaxis]/np.max(I_0h,axis=(0,1))
  I_24h = I_24h[np.newaxis]/np.max(I_24h,axis=(0,1))
  I_36h = I_36h[np.newaxis]/np.max(I_36h,axis=(0,1))
  I_48h = I_48h[np.newaxis]/np.max(I_48h,axis=(0,1))
  
  data = np.stack((I_0h,I_24h,I_36h,I_48h))
  if masked:
    mask = adhesion_mask(data,rscale)
    zs = np.zeros(data.shape)
    data = np.where(np.repeat(np.repeat(mask[np.newaxis],4,axis=0)[:,:,:,:,np.newaxis],4,axis=-1),data,zs)
    return data, mask
  else:
    return data


def adhesion_mask(data,rscale=1.0):
  """
    Given data output from load_sequence_*, returns a binary mask representing the circle where cells can adhere
    
    Parameters
    ----------
    data : float32 array [T,1,size,size,4]
      timesteps (T) of RGBA images. Dummy index of 1 for number of batches

    rscale : float32
      scales how much bigger or smaller the radius of the mask is

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
  
  r*=rscale
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









def periodic_padding(tensor,padding):
    """


    Parameters
    ----------
    tensor : anytype (BATCHES,X,Y,C)
        input 4-tensor
    padding : int 
        how far to pad on X and Y dimensions

    Returns
    -------
    tensor : anytype (BATCHES,X+2*PADDING,Y+2*PADDING,C)
        Periodically padded tensor
    """
    middle = tensor
    right_x = tensor[:,-padding:]
    left_x = tensor[:,:padding]
    tensor_x = tf.concat([right_x,middle,left_x],axis=1)
    
    middle = tensor_x
    right_xy = tensor_x[:,:,-padding:]
    left_xy = tensor_x[:,:,:padding]
    tensor_xy = tf.concat([right_xy,middle,left_xy],axis=2)
    return tensor_xy
    