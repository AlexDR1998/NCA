import jax
import equinox as eqx
import jax.numpy as jnp
import time
import diffrax
from PDE.reaction_diffusion_advection.update import F
from pathlib import Path
from typing import Union

class PDE_solver(eqx.Module):
	func: F	
	dt0: float
	def __init__(self,N_CHANNELS,PERIODIC,dx=0.1,dt=0.1,key=jax.random.PRNGKey(int(time.time()))):
		self.func = F(N_CHANNELS,PERIODIC,dx,key)
		self.dt0 = dt
	def __call__(self, ts, y0):
		solution = diffrax.diffeqsolve(diffrax.ODETerm(self.func),
									   diffrax.Tsit5(),
									   t0=ts[0],t1=ts[-1],
									   dt0=self.dt0,
									   y0=y0,
									   max_steps=16**4,
									   stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
									   saveat=diffrax.SaveAt(ts=ts))
		return solution.ts,solution.ys
	
	def partition(self):
		func_diff,func_static = self.func.partition()
		where = lambda s:s.func
		total_diff,total_static = eqx.partition(self,eqx.is_array)
		total_diff = eqx.tree_at(where,total_diff,func_diff)
		total_static=eqx.tree_at(where,total_static,func_static)
		return total_diff,total_static
		
	def combine(self,diff,static):
		self = eqx.combine(diff,static)
	

	def save(self, path: Union[str, Path], overwrite: bool = False):
		"""
		Wrapper for saving NCA via pickle. Taken from https://github.com/google/jax/issues/2116

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
		suffix = ".eqx"
		path = Path(path)
		if path.suffix != suffix:
			path = path.with_suffix(suffix)
			path.parent.mkdir(parents=True, exist_ok=True)
		if path.exists():
			if overwrite:
				path.unlink()
			else:
				raise RuntimeError(f'File {path} already exists.')
		eqx.tree_serialise_leaves(path,self)
		#with open(path, 'wb') as file:	
			#pickle.dump(self, file)
	
	def load(self, path: Union[str, Path]):
		"""
		

		Parameters
		----------
		path : Union[str, Path]
			path to filename.

		Raises
		------
		ValueError
			Not a file or incorrect file type.

		Returns
		-------
		NCA
			NCA loaded from pickle.

		"""
		suffix = ".eqx"
		path = Path(path)
		if not path.is_file():
			raise ValueError(f'Not a file: {path}')
		if path.suffix != suffix:
			raise ValueError(f'Not a {suffix} file: {path}')
		#with open(path, 'rb') as file:
		#	data = pickle.load(file)
		return eqx.tree_deserialise_leaves(path,self)
		