# Program runs slowly, needs optimization, only good for demonstration
# Range of dimensions can be decreased for faster result
import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

N = int(1e6) # number of points
d = range(1,21)
r = 1

def curseOfDim(N,d,r):
    cube = np.random.uniform(0,1, size=(N,d)) #points in hypercube
    inSphere = 0 #number of points inside the sphere

    for i in range(N):
        if euclidean(np.zeros((1,d)), cube[i]) < r: 
            inSphere += 1
    return inSphere / float(N) #ratio of points
    


listOfRatios = []
for k in d:
    ratio = curseOfDim(N,k,r)
    print("Dimension = {}".format(k))
    print("Volume of hypercube = {}".format((2*r)**k))
    print("Ratio = {}".format(ratio))
    print("*******************")
    listOfRatios.append(ratio)

plt.plot(range(1,len(listOfRatios)+1), listOfRatios)
plt.xlabel("Dimension")
plt.ylabel("Hypershpere / Hypercube")
plt.grid()
plt.show()


    