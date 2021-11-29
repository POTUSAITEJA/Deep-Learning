import cv2
import numpy as np
import matplotlib.pyplot as plt

avgen = cv2.imread("img/day3.jpg")
avgen = cv2.cvtColor(avgen,cv2.COLOR_BGR2RGB)

green = cv2.imread("img/lady.jpg")
hsv = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)

avgen = cv2.resize(avgen,(480,270))
green = cv2.resize(green,(480,270))

lower_bounds = (40,0,30)
upper_bounds =(80,255,255)
 
mask = cv2.inRange(hsv,lower_bounds,upper_bounds)

img = green.copy()
img[mask==255]=[0, 0, 0]
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

back = avgen.copy()
back[mask==0] = [0,0,0]

final = img+back
rgb_final_img = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)


title = ["Background Image","green screen image","hsv","hsv_mask","back_mask","Final Image"]
images = [avgen,green,mask,img,back,final]
for i in range (6):
    plt.subplot(2,3,i+1),plt.imshow(images[i])
    plt.title(title[i])
    plt.xticks([]),plt.yticks([])

plt.show()