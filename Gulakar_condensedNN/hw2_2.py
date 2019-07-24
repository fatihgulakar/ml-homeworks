from imblearn.under_sampling import CondensedNearestNeighbour
import numpy as np
from data import X,y
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#FFFFE0'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#9B870C'])

for data in X:
    yind = X.index(data)
    yt = y[yind]
    
    X_train,X_test,y_train,y_test = train_test_split(data,yt,test_size=0.2,random_state=1,stratify=yt)

    cnn = CondensedNearestNeighbour()
    Xc,yc = cnn.fit_sample(data,yt)
    X_train_cnn,X_test_cnn,y_train_cnn,y_test_cnn = train_test_split(data,yt,test_size=0.2,random_state=1,stratify=yt)

    clf1 = neighbors.KNeighborsClassifier(n_neighbors=1)
    clf1.fit(X_train,y_train)
    pred1 = clf1.predict(X_test)
    pred_cnn1 = clf1.predict(X_test_cnn)

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))

    x_minc, x_maxc = Xc[:, 0].min() - 1, Xc[:, 0].max() + 1
    y_minc, y_maxc = Xc[:, 1].min() - 1, Xc[:, 1].max() + 1
    xxc, yyc = np.meshgrid(np.arange(x_minc, x_maxc, 0.02),np.arange(y_minc, y_maxc, 0.02))

    # for standard k-NN plot
    Z1 = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)

    Z2 =clf1.predict(np.c_[xxc.ravel(), yyc.ravel()])
    Z2 = Z2.reshape(xxc.shape)

    acc = accuracy_score(y_test,pred1)
    acc_cnn = accuracy_score(y_test_cnn,pred_cnn1)

    plt.figure()
    plt.subplot(1,2,1)
    plt.pcolormesh(xx, yy, Z1, cmap=cmap_light)
    plt.scatter(data[:, 0], data[:, 1], c=yt, cmap=cmap_bold,edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('KNN Classifier k = 1, weights Uniform, metric Euclidean Accuracy = {:2f}'.format(acc))

    plt.subplot(1,2,2)
    plt.pcolormesh(xxc, yyc, Z2, cmap=cmap_light)
    plt.scatter(Xc[:, 0], Xc[:, 1], c=yc, cmap=cmap_bold,edgecolor='k', s=20)
    plt.xlim(xxc.min(), xxc.max())
    plt.ylim(yyc.min(), yyc.max())
    plt.title('KNN Classifier k=1, weights Uniform, metric Euclidean, CNN Randomly chosing Accuracy = {:2f}'.format(acc_cnn))

plt.show()