import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

imgDir = '/home/fatih/codes/pca_hw'
images = [os.path.join(imgDir, f) for f in os.listdir(imgDir) if f.endswith(".jpg")]

immatrix = np.array([np.array(Image.open(im,'r')).flatten() for im in images], 'f')
pca = PCA(0.85)

pca_images = pca.fit_transform(immatrix)
approximation = pca.inverse_transform(pca_images)

plt.figure()
plt.suptitle("Original images")
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(immatrix[i].reshape(384,384), cmap='Greys_r', interpolation='nearest',clim=(0,255))
    plt.axis('off')

plt.figure()
plt.suptitle("PCA approximation images - 85% of principal components")
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(approximation[i].reshape(384,384), cmap='Greys_r', interpolation='nearest',clim=(0,255))
    plt.axis('off')
    
plt.show()

pca = PCA()
pca.fit(immatrix)
total = sum(pca.explained_variance_)
var_exp = [(i/total)*100 for i in sorted(pca.explained_variance_,reverse=True)]
cum_var_exp = np.cumsum(var_exp)
componentsVariance = [25, np.argmax(cum_var_exp > 99) + 1, np.argmax(cum_var_exp > 95) + 1, np.argmax(cum_var_exp > 90) + 1, np.argmax(cum_var_exp >= 85) + 1]
mean_face = np.zeros(immatrix[0].shape)
for i in immatrix:
    mean_face = np.add(mean_face,i)
mean_face = np.divide(mean_face, 25).flatten()
plt.figure()
plt.imshow(mean_face.reshape(384,384),cmap='Greys_r')
plt.show()


plt.figure()
plt.step(range(1, 26), cum_var_exp, where='mid',label='cumulative explained variance')
plt.title('Cumulative Explained Variance as a Function of the Number of Components')
plt.ylabel('Cumulative Explained variance')
plt.xlabel('Principal components')
plt.axhline(y = 95, color='k', linestyle='--', label = '95% Explained Variance')
plt.axhline(y = 90, color='c', linestyle='--', label = '90% Explained Variance')
plt.axhline(y = 85, color='r', linestyle='--', label = '85% Explained Variance')
plt.legend(loc='best')
plt.show()

def explainedVar(perc,images):
    pca = PCA(perc)
    pca.fit(images)
    comp = pca.transform(images)
    approx = pca.inverse_transform(comp)
    return approx

plt.subplot(1,5,1)
plt.imshow(immatrix[16].reshape(384,384),cmap='Greys_r',interpolation='nearest',clim=(0,255))
plt.xlabel("25 components")
plt.title("Original image")

plt.subplot(1,5,2)
plt.imshow(explainedVar(0.99,immatrix)[16].reshape(384,384),cmap='Greys_r',interpolation='nearest',clim=(0,255))
plt.xlabel("22 components")
plt.title("99% variance")

plt.subplot(1,5,3)
plt.imshow(explainedVar(0.95,immatrix)[16].reshape(384,384),cmap='Greys_r',interpolation='nearest',clim=(0,255))
plt.xlabel("17 components")
plt.title("95% variance")

plt.subplot(1,5,4)
plt.imshow(explainedVar(0.9,immatrix)[16].reshape(384,384),cmap='Greys_r',interpolation='nearest',clim=(0,255))
plt.xlabel("14 components")
plt.title("90% variance")

plt.subplot(1,5,5)
plt.imshow(explainedVar(0.85,immatrix)[16].reshape(384,384),cmap='Greys_r',interpolation='nearest',clim=(0,255))
plt.xlabel("11 components")
plt.title("85% variance")

plt.show()