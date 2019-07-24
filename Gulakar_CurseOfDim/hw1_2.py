import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from math import sqrt

N = int(1e6)
d = range(1,41)
means = []
stds = []
distances = []
for dim in d:
    for experiment in range(1,31):
        cube = np.random.uniform(0,1,size=(N,dim))
        point1 = cube[np.random.randint(1,N)]
        point2 = cube[np.random.randint(1,N)]
        distances.append(euclidean(point1,point2))
    means.append(np.mean(distances))
    stds.append(np.std(distances))
    distances = []
means = np.array(means)
stds = np.array(stds)
coeffVariation = np.divide(stds,means)

plt.subplot(3,1,1)
plt.plot(range(1,len(d)+1),stds,'b:')
plt.ylabel("Standard Deviation")
plt.xlabel("Dimensions")

plt.subplot(3,1,2)
plt.plot(range(1,len(d)+1),stds,'r-.')
plt.ylabel("Mean")
plt.xlabel("Dimensions")

plt.subplot(3,1,3)
plt.plot(range(1,len(d)+1),coeffVariation,'g--')
plt.ylabel('Coefficient of Variation')
plt.xlabel("Dimensions")

plt.grid()
plt.show()
