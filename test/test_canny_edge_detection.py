import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('D:/DataSets/TableBank/Recognition/images/%c3%89pid%c3%a9miologie%20du%20Diab%c3%a8te+PNL)_1.png', 0)
edges = cv.Canny(img, 100, 200)
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
