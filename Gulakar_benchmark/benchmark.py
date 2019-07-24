import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import random

iteration = 6
centers = [[-8,8], [0,8], [8,8], [-8,0], [0,0], [8,0], [-8,-8], [0,-8],[8,-8]]
centers_random = np.array([(random.random()*8.0, random.random()*8.0) for _ in range(9)])
X, y = make_blobs(n_samples=150, centers=centers, cluster_std=0.85)

dbs_kpp = []
silcoeff_kpp = []
dbs_frandom = []
silcoeff_frandom = []
dbs_forgy = []
silcoeff_forgy = []

# Random partition part
y_randomp = np.random.randint(low=0, high=8, size=150)
# Could not calculate centroids :(

plt.figure()
plt.suptitle("Each row is the next iteration")
for i in range(1,iteration):
    km_kpp = KMeans(n_clusters=9,init='k-means++', max_iter=1)
    km_frandom = KMeans(n_clusters=9, init=centers_random, max_iter=1)
    km_forgy = KMeans(n_clusters=9, init='random', max_iter=1)

    km_kpp.fit(X)
    km_frandom.fit(X)
    km_forgy.fit(X)

    y_kpp = km_kpp.predict(X)
    y_frandom = km_frandom.predict(X)
    y_forgy = km_forgy.predict(X)
    
    dbs_kpp.append(davies_bouldin_score(X,y_kpp))
    silcoeff_kpp.append(silhouette_score(X,y_kpp))

    dbs_frandom.append(davies_bouldin_score(X,y_frandom))
    silcoeff_frandom.append(silhouette_score(X,y_frandom))

    dbs_forgy.append(davies_bouldin_score(X,y_forgy))
    silcoeff_forgy.append(silhouette_score(X,y_forgy))

    plt.subplot(iteration-1, 3, 3*i-2)
    plt.title("kmeans++")
    plt.scatter(X[:,0],X[:,1], marker='o', c=y_kpp, s=50, cmap='viridis')
    center_kpp = km_kpp.cluster_centers_
    plt.scatter(center_kpp[:,0], center_kpp[:,1], c='black', s=200, alpha=0.5)

    plt.subplot(iteration-1, 3, 3*i - 1)
    plt.title("Fully random")
    plt.scatter(X[:,0],X[:,1], marker='o', c=y_frandom, s=50, cmap='viridis')
    center_frandom = km_frandom.cluster_centers_
    plt.scatter(center_frandom[:,0], center_frandom[:,1], c='black', s=200, alpha=0.5)

    plt.subplot(iteration-1, 3, 3*i)
    plt.title("Forgy")
    plt.scatter(X[:,0],X[:,1], marker='o', c=y_forgy, s=50, cmap='viridis')
    center_forgy = km_forgy.cluster_centers_
    plt.scatter(center_forgy[:,0], center_forgy[:,1], c='black', s=200, alpha=0.5)

    # update centers for fully random
    centers_random = center_frandom

plt.show()

plt.subplot(3,2,1)
plt.title("David-Bouldain Score, Kmeans++")
error1 = np.std(dbs_kpp)
plt.errorbar(range(1,iteration), dbs_kpp, xerr=error1, errorevery=1, markeredgewidth=20)
plt.grid()

plt.subplot(3,2,3)
plt.title("David-Bouldain Score, Fully Random")
error3 = np.std(dbs_frandom)
plt.errorbar(range(1,iteration), dbs_frandom, xerr=error3, errorevery=1, markeredgewidth=20)
plt.grid()

plt.subplot(3,2,5)
plt.title("David-Bouldain Score, Forgy")
error5 = np.std(dbs_forgy)
plt.errorbar(range(1,iteration), dbs_forgy, xerr=error5, errorevery=1, markeredgewidth=20)
plt.grid()

plt.subplot(3,2,2)
plt.title("Silhouette Coefficient, Kmeans++")
error2 = np.std(silcoeff_kpp)
plt.errorbar(range(1,iteration), silcoeff_kpp, xerr=error2, errorevery=1, markeredgewidth=20)
plt.grid()

plt.subplot(3,2,4)
plt.title("Silhouette Coefficient, Fully random")
error4 = np.std(silcoeff_frandom)
plt.errorbar(range(1,iteration), silcoeff_frandom, xerr=error4, errorevery=1, markeredgewidth=20)
plt.grid()

plt.subplot(3,2,6)
plt.title("Silhouette Coefficient, Forgy")
error6 = np.std(silcoeff_forgy)
plt.errorbar(range(1,iteration), silcoeff_forgy, xerr=error6, errorevery=1, markeredgewidth=20)
plt.grid()

plt.show()

