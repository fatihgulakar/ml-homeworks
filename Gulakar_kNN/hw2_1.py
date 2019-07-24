import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.neighbors import KNeighborsClassifier,DistanceMetric
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from data import X,y

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#FFFFE0','#EE82EE'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#9B870C','#BA55D3'])

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = KNeighborsClassifier() #k=5
clf3 = KNeighborsClassifier(weights='distance') #k=5

for data in X:
    yind = X.index(data)
    yt = y[yind]
    X_train,X_test,y_train,y_test = train_test_split(data,yt,test_size=0.2,random_state=1,stratify=yt)
    
    clf4 = KNeighborsClassifier(n_neighbors=1,metric='mahalanobis',metric_params={'V': np.cov(X_train,rowvar=False)})

    clf1.fit(X_train,y_train)
    clf2.fit(X_train,y_train)
    clf3.fit(X_train,y_train)
    clf4.fit(X_train,y_train)

    predicted1 = clf1.predict(X_test)
    predicted2 = clf2.predict(X_test)
    predicted3 = clf3.predict(X_test)
    predicted4 = clf4.predict(X_test)

    acc1 = accuracy_score(y_test,predicted1)
    acc2 = accuracy_score(y_test,predicted2)
    acc3 = accuracy_score(y_test,predicted3)
    acc4 = accuracy_score(y_test,predicted4)

    x_min,x_max = data[:,0].min() - 1, data[:,0].max() + 1
    y_min,y_max = data[:,1].min() - 1, data[:,1].max() + 1
    xx,yy = np.meshgrid(np.arange(x_min,x_max,0.02), np.arange(y_min,y_max,0.02))
    Z1 = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)
    Z2 = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
    Z2 = Z2.reshape(xx.shape)
    Z3 = clf3.predict(np.c_[xx.ravel(), yy.ravel()])
    Z3 = Z3.reshape(xx.shape)
    Z4 = clf4.predict(np.c_[xx.ravel(), yy.ravel()])
    Z4 = Z4.reshape(xx.shape)
    plt.figure()
    plt.subplot(2,2,1)
    plt.pcolormesh(xx,yy,Z1,cmap=cmap_light)
    plt.scatter(data[:,0], data[:,1], c=yt,cmap=cmap_bold,edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("k=1, majority voting, Euclidean metric Accuracy = {:2f}".format(acc1))

    plt.subplot(2,2,2)
    plt.pcolormesh(xx,yy,Z2,cmap=cmap_light)
    plt.scatter(data[:,0], data[:,1], c=yt,cmap=cmap_bold,edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("k=5, majority voting, Euclidean metric Accuracy = {:2f}".format(acc2))

    plt.subplot(2,2,3)
    plt.pcolormesh(xx,yy,Z3,cmap=cmap_light)
    plt.scatter(data[:,0], data[:,1], c=yt,cmap=cmap_bold, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("k=5, distance voting, Euclidean metric Accuracy = {:2f}".format(acc3))

    plt.subplot(2,2,4)
    plt.pcolormesh(xx,yy,Z4,cmap=cmap_light)
    plt.scatter(data[:,0], data[:,1], c=yt,cmap=cmap_bold,edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("k=1, majority voting, Mahalanobis metric Accuracy = {:2f}".format(acc4))
    
plt.show()