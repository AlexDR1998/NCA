import jax
import equinox as eqx
import jax.numpy as jnp
import time
import diffrax
from PDE.reaction_diffusion_advection.update import F

class PDE_solver(eqx.Module):
	func: F	
	def __init__(self,N_CHANNELS,PERIODIC,dx=0.1,key=jax.random.PRNGKey(int(time.time()))):
		self.func = F(N_CHANNELS,PERIODIC,dx,key)
		
	def __call__(self, ts, y0):
		solution = diffrax.diffeqsolve(diffrax.ODETerm(self.func),
									   diffrax.Tsit5(),
									   t0=ts[0],t1=ts[-1],
									   dt0=ts[1] - ts[0],
									   y0=y0,
									   stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
									   saveat=diffrax.SaveAt(ts=ts))
		return solution.ys,solution.ts
	
	def partition(self):
		func_diff,func_static = self.func.partition()
		where = lambda s:s.func
		total_diff,total_static = eqx.partition(self,eqx.is_array)
		total_diff = eqx.tree_at(where,total_diff,func_diff)
		total_static=eqx.tree_at(where,total_static,func_static)
		return total_diff,total_static
		
	def combine(self,diff,static):
		self = eqx.combine(diff,static)