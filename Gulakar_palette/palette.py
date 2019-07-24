import matplotlib.pyplot as plt
from scipy.misc import imread, imshow
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.image as image
from sklearn.decomposition import PCA

img = imread('color.jpg')

X = (img/255.0).reshape(-1,3)
dbs = []
for i in range(2,64):   
    kmean = KMeans(n_clusters=i)
    kmean.fit(X)
    y = kmean.predict(X)
    dbs.append(davies_bouldin_score(X,y))

plt.figure()
plt.plot(range(2,64), dbs,'r--')
plt.xlabel("Number of clusters")
plt.ylabel("David-Bouldin Score")
plt.grid()

plt.show()

# k_colors = KMeans(n_clusters=128).fit(X)
# img128=k_colors.cluster_centers_[k_colors.labels_]
# img128=np.reshape(img128, (img.shape))
# image.imsave('img128.png',img128)

