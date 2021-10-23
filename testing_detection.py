import numpy as np
import cv2
from PIL import Image, ImageFilter

original_img = cv2.imread("00006.jpg")
cv2.imshow("Original image", original_img)

img = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
#cv2.imshow("HSV image", img)

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow("image grayscale", img_gray)

ret, thresh = cv2.threshold(img_gray, 55, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("thresh", thresh)

# remove white spaces inside the bounding box using erosion
kernel = np.ones([10,10])
eroded_image = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
cv2.imshow("Erosion image", eroded_image)

# rectangle detection
contours, _ = cv2.findContours(eroded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print("number of contours detected: {}".format(len(contours)))
index=0
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0 * cv2.arcLength(contour, True), True)
    cv2.drawContours(original_img, [approx], 0, (255 , 0, 0), 3)
    index=index+1
    print("How many lines detected in contour #{}: {}".format(index,len(approx)))
    x, y , w, h = cv2.boundingRect(approx)
    print("x_pos={}, y_pos={}, width={}, height={}".format(x, y, w, h))

cv2.imshow("with contours", original_img)
cv2.waitKey(0)