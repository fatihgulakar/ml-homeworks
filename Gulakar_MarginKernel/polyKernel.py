import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import itertools
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


X,y = make_classification(n_samples=120, n_features=2,n_informative=2, n_redundant=0,random_state=54,flip_y=0.1)

#plt.figure()
#plt.scatter(X[:,0],X[:,1], marker='o', c=y, s=50, cmap='viridis')
#plt.show()

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def plot_boundaries(ax, clf, xx, yy, **params):
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = clf.decision_function(xy).reshape(xx.shape)
    out = ax.contour(xx, yy, Z, **params)
    return out

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

Clist = {0.1, 1, 5}
gammas = {0.001, 0.01, 0.9}

i = 0
for x in itertools.product(gammas,Clist):
    i = i+1
    clf = SVC(C= x[1],kernel='poly',gamma = x[0])
    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)
    score = clf.score(X,y)
    acc = accuracy_score(y_test,y_pred)
    # plt.subplot(3,3,i)
    # plt.title("C = {:1f}, gamma = {:2f}, accuracy={:2f}".format(x[1], x[0], acc))
    # plt.scatter(X_test[:,0],X_test[:,1], marker='o', c=y_pred, s=50, cmap='viridis')

    fig, ax = plt.subplots()
    plt.suptitle("C={}, gamma={}".format(x[1],x[0]))
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_boundaries(ax, clf, xx, yy, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])

    plot_contours(ax, clf, xx, yy, cmap=cm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=cm_bright, s=40)
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s = 40,
            linewidth=1.5, facecolors='none', edgecolors = 'g', label='Support Vectors')
    

    ax.text(xx.max() - .3, yy.min() + .3, ('%.4f' % acc).lstrip('0'),
                size=15, horizontalalignment='right')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.legend()


plt.show()



