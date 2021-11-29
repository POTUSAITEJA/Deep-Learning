import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range 
of edges in images. It was developed by John F. Canny in 1986. Canny also produced a computational theory of edge 
detection explaining why the technique works.

The Canny edge detection algorithm is composed of 5 steps:

1. Noise reduction;
2. Gradient calculation;
3. Non-maximum suppression;
4. Double threshold;
5. Edge Tracking by Hysteresis.

'''

car = cv2.imread("img/car.jpg",cv2.IMREAD_GRAYSCALE)
lap = cv2.Laplacian(car,cv2.CV_64F,ksize=3)
sobelX = cv2.Sobel(car,cv2.CV_64F,1,0)
sobelY = cv2.Sobel(car,cv2.CV_64F,0,1)
canny = cv2.Canny(car,10,200)

lap = np.uint8(np.absolute(lap))
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))

sobel_combined = cv2.bitwise_or(sobelX,sobelY)


title = ["images","Laplacian","sobelX","sobelY","sbel_combined","canny"]
images = [car,lap,sobelX,sobelY,sobel_combined,canny]
for i in range (6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],"gray")
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])

plt.show()