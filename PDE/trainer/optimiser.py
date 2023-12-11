import optax
import jax
import equinox as eqx



def non_negative_diffusion(learn_rate=1e-2,iters=1000):
	schedule = optax.exponential_decay(1e-2, transition_steps=iters, decay_rate=0.99)
	normal = optax.adamw(schedule)
	only_positive = optax.chain(optax.keep_params_nonnegative,optax.adam(schedule))
	def label_fn(tree):
		# Returns "positive" for the diffusion terms that should remain non-negative
		# Returns "normal" otherwise
		filter_spec = jax.tree_util.tree_map(lambda _:"normal",tree)
		filter_spec = eqx.tree_at(lambda t:t.func.f_d,filter_spec,replace="positive")
		filter_spec = eqx.tree_at(lambda t:t.func.f_v,filter_spec,replace="normal")
		filter_spec = eqx.tree_at(lambda t:t.func.f_r,filter_spec,replace="normal")
		return filter_spec
	return optax.multi_transform({"positive": only_positive,"normal":normal},label_fn)