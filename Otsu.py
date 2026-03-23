import cv2 as cv
import numpy as np

img=cv.imread ("1000016167.jpg")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
h,s,v= cv.split(hsv)
ret_h, th_h = cv.threshold(h,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
ret_s, th_s = cv.threshold(s,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
ret_v, th_v = cv.threshold(s,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imwrite("th_h.png",th_h)
cv.imwrite("th_s.png",th_s)
cv.imwrite("th_v.png",th_v)
