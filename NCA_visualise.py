import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob


"""
	Utilities and helper functions for visualising results. Don't use this on Eddie
"""



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



