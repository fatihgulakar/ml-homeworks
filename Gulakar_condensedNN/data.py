import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles

# Dataset 1
X1,y1 = make_gaussian_quantiles(mean=(7,7),cov=3.,n_samples=40, n_features=2, n_classes=2)
X2,y2 = make_gaussian_quantiles(mean=(-4,4),cov=4.,n_samples=56,n_features=2,n_classes=3)
X3,y3 = make_classification(n_samples=64,n_features=2,n_informative=2,n_redundant=0,n_repeated=0,n_classes=2,n_clusters_per_class=1,class_sep=3,flip_y=0.2,weights=[0.5,0.5],random_state=15)
X4,y4 = make_blobs(n_samples=40,centers=4,n_features=2,random_state=7)
X5,y5 = make_gaussian_quantiles(mean=(-3,-3),cov=2.5,n_samples=40,n_features=2,n_classes=2)
X6,y6 = make_classification(n_samples=60,n_features=2,n_informative=2,n_redundant=0,n_repeated=0,n_classes=2,n_clusters_per_class=1,class_sep=3,flip_y=0.2,weights=[0.5,0.5],random_state=20)

# Dataset 2
X7,y7 = make_blobs(n_samples=24, centers=2, n_features=2, random_state=5)
X8,y8 = make_blobs(n_samples=20, centers=3, n_features=2, random_state=3)
X9,y9 = make_gaussian_quantiles(cov=5, n_samples=28, n_features=2, n_classes=3)
X10,y10 = make_classification(n_samples=22, n_features=2,n_informative=2,n_redundant=0,n_repeated=0,n_classes=2,n_clusters_per_class=1,class_sep=2,flip_y=0.1,weights=[0.3,0.7],random_state=5)

# Dataset 3
X11,y11 = make_gaussian_quantiles(mean=(-10,15),cov=1.5,n_samples=50,n_features=2,n_classes=2)
X12,y12 = make_classification(n_samples=34,n_features=2, n_informative=2,n_redundant=0,n_repeated=0,n_classes=2,n_clusters_per_class=1,class_sep=1,flip_y=0.2,weights=[0.4,0.6], random_state=4)
X13,y13 = make_classification(n_samples=46,n_features=2, n_informative=2,n_redundant=0,n_repeated=0,n_classes=2,n_clusters_per_class=1,class_sep=1,flip_y=0.25,weights=[0.45,0.55], random_state=3)
X14,y14 = make_blobs(n_samples=40,centers=5,n_features=2,random_state=3)
X15,y15 = make_blobs(n_samples=30, centers=2,n_features=2,random_state=8)
Xa = np.concatenate([X1,X2,X3,X4,X5,X6])
ya = np.concatenate([y1,y2,y3,y4,y5,y6])

Xb = np.concatenate([X7,X8,X9,X10])
yb = np.concatenate([y7,y8,y9,y10])

Xc = np.concatenate([X11,X12,X13,X14,X15])
yc = np.concatenate([y11,y12,y13,y14,y15])

X = (Xa,Xb,Xc)
y = (ya,yb,yc)

