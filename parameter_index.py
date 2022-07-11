import numpy as np
import sys

"""
	Takes job array index as input, prints parameters
"""

index = int(sys.argv[1])-1
FIRE_RATE_INTS = (np.arange(10).astype(int)+1)*2
DECAY_FACTOR_INTS= (np.arange(10).astype(int)+1)*2
N_CHANNELS = np.arange(5,10)
#print(FIRE_RATE_INTS)

x = (index//10)%10
y = index%10
z = (index//100)%10

print(FIRE_RATE_INTS[x])
print(DECAY_FACTOR_INTS[y])
print(N_CHANNELS[z])
