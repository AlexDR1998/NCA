import numpy as np
import sys

"""
	Takes job array index as input, prints parameters
"""


index = int(sys.argv[1])-1 	# index between 1 and X_PARAMS.shape[0]*Y_PARAMS.shape[0]*Z_PARAMS.shape[0]
X_PARAMS = np.arange(10)  	# Arrays of parameter values - edit
Y_PARAMS = np.arange(5)  	# Arrays of parameter values - edit
Z_PARAMS = np.arange(2)  	# Arrays of parameter values - edit


x = index%X_PARAMS.shape[0]
y = (index//X_PARAMS.shape[0])%Y_PARAMS.shape[0]
z = (index//(X_PARAMS.shape[0]*Y_PARAMS.shape[0]))%Z_PARAMS.shape[0]

print(X_PARAMS[x])
print(Y_PARAMS[y])
print(Z_PARAMS[z])
