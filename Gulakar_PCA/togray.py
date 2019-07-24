import cv2
import glob, os, errno

mydir = '/home/fatih/codes/pca_hw/images/crop/grayscales'
try:
    os.makedirs(mydir)
except OSError as e:
    if e.errno == errno.EEXIST:
        raise
for fil in glob.glob("*.jpg"):
    image = cv2.imread(fil)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(mydir,fil),gray_image)