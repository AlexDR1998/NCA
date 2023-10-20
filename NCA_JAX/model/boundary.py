import jax.numpy as jnp
import jax
import equinox as eqx

class NCA_boundary(object):
	"""
		Callable object that forces intermediate NCA states to be fixed to boundary condition at specified channels
	"""
	
	
	def __init__(self,mask = None):
		"""
		Parameters
		----------
		mask : float32 [MASK_CHANNELS,WIDTH,HEIGHT]
			array encoding structure or boundary conditions for NCA intermediate states
		Returns
		-------
		None.

		"""

		self.MASK = mask
		
	#@eqx.filter_jit	
	def __call__(self,x):
		if self.MASK is None:
			return x
		else:
			m_channels = self.MASK.shape[0]
			#print(self.MASK.shape)
			x_masked = x.at[-m_channels:].set(self.MASK)
			return x_masked