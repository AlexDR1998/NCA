import jax
import jax.numpy as jnp

print(jax.default_backend())
print(jax.devices())
key = jax.random.PRNGKey(0)

A = jax.random.uniform(key,(2,500,500))
key = jax.random.fold_in(key,1)
B = jax.random.uniform(key,(2,500,500))

def matmul(a,b):
	return jnp.einsum("ij,jk->ik",a,b)

vmat = jax.vmap(matmul)
pmat = jax.pmap(matmul)


shard = jax.sharding.PmapSharding(A.shape,sharded_dim=0)
print(shard)
A_dev,B_dev = jax.device_put((A,B),shard)

C_dev = pmat(A_dev,B_dev)
C = vmat(A,B)
print(C)
print(C_dev)
