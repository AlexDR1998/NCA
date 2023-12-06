from NCA_JAX.trainer.data_augmenter_tree import DataAugmenter

class DataAugmenterPDE(DataAugmenter):
	"""
		Inherits the methods of DataAugmenter, but overwrites the batch cloning in the init
	"""
	def data_init(self,SHARDING=None):
		return None
	
	
	