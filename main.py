import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap= cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
segmentor= SelfiSegmentation()
imgbg= cv2.imread("bg5.png")

while True:
    success, img= cap.read()
    imgout= segmentor.removeBG(img, imgbg, threshold= .55555)
    #imgout= cv2.cvtColor(imgout, cv2.COLOR_GRAY2RGB)

    imgstacked= cvzone.stackImages([img, imgout], 2, 1)
    cv2.imshow("Output Image", imgout)
    cv2.waitKey(1)