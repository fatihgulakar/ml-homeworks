# Code works very slow since dimension and number of experiments are high
# They can be reduced for faster result
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt

N = int(1e6)
d = range(1,101)
means = []
for dim in d:
    for experiment in range(1,21):
        angles = []
        cube = np.random.uniform(0,1,size=(N,dim))
        point1 = cube[np.random.randint(1,N)]
        point2 = cube[np.random.randint(1,N)]
        point3 = cube[np.random.randint(1,N)]
        point4 = cube[np.random.randint(1,N)]

        vec1 = np.array(point1) - np.array(point2)
        vec2 = np.array(point3) - np.array(point4)
        dist = spatial.distance.cosine(vec1,vec2)
        phi = 1-dist
        if not (-1 <= phi <= 1):
            if phi >= 1:
                phi = 0
            if phi <= -1:
                phi = 180
        angles.append( np.degrees(np.arccos(phi)) )
    means.append(np.mean(angles))

plt.plot(range(1,len(d) + 1),means,'r*')
plt.plot([90]*len(means),'--')
plt.xlabel("Dimensions")
plt.ylabel("Average angle in degrees")
plt.grid()
plt.show()
    