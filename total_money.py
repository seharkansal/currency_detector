import cv2
import numpy as np
# img=cv2.imread("indian.png")
# output=img.copy()
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray_blur=cv2.medianBlur(gray,5)
# circles=cv2.HoughCircles(gray_blur,cv2.HOUGH_GRADIENT,1.55,110)
coins = cv2.imread('indian.png', 1)

gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(gray, 7)
output=coins.copy()
circles = cv2.HoughCircles(
img,  # source image
cv2.HOUGH_GRADIENT,  # type of detection
        1,
        50,
        param1=100,
        param2=50,
        minRadius=10,  # minimal radius
        maxRadius=380,  # max radius
    )

dect_circles= np.round(circles[0, :]).astype("int")
for (x,y,r) in dect_circles:
    cv2.circle(output,(x,y),r,(0,255,0),3)
    cv2.circle(output,(x,y),2,(0,255,255),3)


cv2.imshow('output',output)
cv2.waitKey(0)
cv2.destroyAllWindows()