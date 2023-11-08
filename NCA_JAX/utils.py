import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import skimage
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
import os
import scipy as sp
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm
from tensorflow.python.framework import tensor_util
from pathlib import Path
from typing import Union
import pickle
import tensorflow as tf
# Some convenient helper functions
#@jax.jit


def save_pickle(data, path: Union[str, Path], overwrite: bool = False):
    """
    Taken from https://github.com/google/jax/issues/2116

    Parameters
    ----------
    path : Union[str, Path]
        path to filename.
    overwrite : bool, optional
        Overwrite existing filename. The default is False.

    Raises
    ------
    RuntimeError
        file already exists.

    Returns
    -------
    None.

"""
    suffix = ".pickle"
    path = Path(path)
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f'File {path} already exists.')
    with open(path, 'wb') as file:
        pickle.dump(data, file)
    
def load_pickle(path: Union[str, Path]):
    
    suffix = '.pickle'
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Not a file: {path}')
    if path.suffix != suffix:
        raise ValueError(f'Not a {suffix} file: {path}')
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data






def key_array_gen(key,shape):
	"""
	Parameters
	----------
	key : jax.random.PRNGKey, 
		Jax random number key.
	shape : tuple of ints
		Shape to broadcast to

	Returns
	-------
	key_array : uint32[shape,2]
		array of random keys
	"""
	shape = list(shape)
	shape.append(2)
	key_array = jax.random.randint(key,shape=shape,minval=0,maxval=2_147_483_647,dtype="uint32")
	return key_array

def key_pytree_gen(key,shape):
	"""
	
	
	Parameters
	----------
	key : jax.random.PRNGKey, 
		Jax random number key.
	shape : tuple of ints
		Shape to broadcast to

	Returns
	-------
	key_array : uint32[shape,2]
		array of random keys
	"""
	shape = list(shape)
	shape.append(2)
	key_array = jax.random.randint(key,shape=shape,minval=0,maxval=2_147_483_647,dtype="uint32")
	key_array = list(key_array)
	return key_array

#def key_array_gen_pytree(key,BATCHES,N):
#	key_array = []
#	for i in range(BATCHES):
		


def grad_norm(grad):
	"""
	Normalises each vector/matrix in grad 

	Parameters
	----------
	grad : NCA/pytree

	Returns
	-------
	grad : NCA/pytree

	"""
	w_where = lambda l: l.weight
	b_where = lambda l: l.bias
	w1 = grad.layers[3].weight/(jnp.linalg.norm(grad.layers[3].weight)+1e-8)
	w2 = grad.layers[5].weight/(jnp.linalg.norm(grad.layers[5].weight)+1e-8)
	b2 = grad.layers[5].bias/(jnp.linalg.norm(grad.layers[5].bias)+1e-8)
	grad.layers[3] = eqx.tree_at(w_where,grad.layers[3],w1)
	grad.layers[5] = eqx.tree_at(w_where,grad.layers[5],w2)
	grad.layers[5] = eqx.tree_at(b_where,grad.layers[5],b2)
	return grad

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
		images : float32 array [1,T,C,size,size]
			Timesteps of T RGB/RGBA images. Dummy index of 1 for number of batches
	"""
	images = []
	for filename in filename_sequence:
		im = skimage.io.imread(impath_emojis+filename)[::downsample,::downsample]
		if crop_square:
			s= min(im.shape[0],im.shape[1])
			im = im[:s,:s]
		#im = im[np.newaxis] / 255.0
		im = im/255.0
		images.append(im)
	data = np.array(images)
	data = data[np.newaxis]
	data = np.einsum("btxyc->btcxy",data)
	return data

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



def load_micropattern_radii(impath):
	filenames = glob.glob(impath)
	filenames = list(sorted(filenames))
	#print(sorted(filenames))
	ims = []
	for f_str in filenames:
		ims.append(skimage.io.imread(f_str))
	#print(jax.tree_util.tree_structure(ims))
	
	normalise = lambda arr : arr/np.max(arr,axis=(0,1))
	pad = lambda arr : np.pad(arr,((10,10),(10,10),(0,0)))
	mask_out = lambda arr,mask : np.where(np.repeat(mask[0][:,:,np.newaxis],4,axis=-1),arr,np.zeros_like(arr))
	reshape = lambda arr:np.einsum("xyc->cxy",arr)
	just_mask = lambda mask:mask[0][np.newaxis]
	shapes = lambda arr: arr.shape[-1]
	def stack_x0(arr,mask):
		x0 = np.zeros_like(arr).astype(float)
		masked_arr = np.ma.array(arr,mask=~np.repeat(mask[0][np.newaxis],4,axis=0).astype(bool))
		#print(masked_arr)
		x0[1] = mask[0].astype(x0.dtype)#*masked_arr[1].mean() # Set SOX2 channel to high, everything else is 0
		x0[3] = mask[0].astype(x0.dtype)#*masked_arr[3].mean() # Set LMBR channel to high, everything else is 0
		x0[1]*= masked_arr[1].mean()
		x0[3]*= masked_arr[3].mean()
		return np.stack((x0,arr),axis=0)
	ims = list(map(lambda x: pad(normalise(x)),ims))
	masks = list(map(adhesion_mask_convex_hull,tqdm(ims)))
	ims = list(map(mask_out,ims,masks))
	ims = list(map(reshape,ims))
	ims = list(map(stack_x0,ims,masks))
	masks = list(map(just_mask,masks))
	shapes = list(map(shapes,ims))
		
	#plt.hist(shapes)
	#plt.show()

# 	for i in range(len(shapes)):
# 		xx = masks[i][1]
# 		yy = masks[i][2]
# 		rs = masks[i][3]
# 		convex = masks[i][4]
# 		plt.hist(ims[i][:,:,0].flatten(),alpha=0.5,bins=50,label="SOX17")
# 		plt.hist(ims[i][:,:,1].flatten(),alpha=0.5,bins=50,label="SOX2")
# 		plt.hist(ims[i][:,:,2].flatten(),alpha=0.5,bins=50,label="TBXT")
# 		plt.hist(ims[i][:,:,3].flatten(),alpha=0.5,bins=50,label="LMBR")
# 		#print("Minimum: "+str(np.min(ims[i])))
# 		#print("Maximum: "+str(np.max(ims[i])))
# 		
# 		plt.legend()
# 		plt.show()
# 		fig,ax = plt.subplots(1)
# 		circ = Circle((xx,yy),rs,fill=False,color="yellow")
# 		ax.add_patch(circ)
# 		ax.imshow(ims[i][:,:,:3])
# 		plt.show()
# 		plt.imshow(masks[i][0])
# 		plt.show()
# 		plt.imshow(convex)
# 		plt.show()
# 		#plt.imshow(masks[i]*ims[i][:,:,3])
# 		
# 		#plt.show()
# 		#print(masks)
# 	print(shapes)
	#print(ims.shape)
	
	
	#ims = jax.tree_util.treedef_tuple(ims)
	#print(jnp.mean(ims))
	return ims,masks,shapes




def load_sequence_A(name,impath):
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

def load_sequence_B(name,impath):
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

def load_sequence_ensemble_average(impath,masked=True,rscale=1.0):
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

  #thresh = data[...,-1]# use LMBR staining?
  
  thresh = np.mean(data,axis=-1) 
  thresh = sp.ndimage.gaussian_filter(thresh,5)
  
  
  k = thresh>np.mean(thresh)
  
  regions = skimage.measure.regionprops(skimage.measure.label(k))
  cell_culture = regions[0]

  x0, y0 = cell_culture.centroid
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
  return mask,x0,y0,r

def adhesion_mask_convex_hull(data,rscale=1.0):
  """
    Given data output from load_sequence_*, returns a binary mask representing the circle where cells can adhere.
    
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
  
  thresh = np.mean(data,axis=-1) 
  thresh = sp.ndimage.gaussian_filter(thresh,1)  
  
  k = thresh>np.mean(thresh)
  k = skimage.morphology.convex_hull_image(k,tolerance=0.1)
  
  regions = skimage.measure.regionprops(skimage.measure.label(k))
  cell_culture = regions[0]

  x0, y0 = cell_culture.centroid
  r = cell_culture.major_axis_length / 2.

  def cost(params):
      x0, y0, r = params
      coords = skimage.draw.disk((y0, x0), r, shape=k.shape)
      template = np.zeros_like(k)
      template[coords] = 1
      return -np.sum(template == k)

  x0, y0, r = sp.optimize.fmin(cost, (x0, y0, r),disp=False)
  mask = np.zeros(k.shape,dtype="float32")
  
  r*=rscale
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      mask[i,j] = (i-y0)**2+(j-x0)**2<r**2
  #print(mask.shape)
  return mask,x0,y0,r,k
  #return mask

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

