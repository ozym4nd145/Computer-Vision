import numpy as np
import cv2
import mog

THRESHOLD = 30
CONFIDENCE = 0.1

cap = cv2.VideoCapture('./videos/vtest.avi')
#cap = cv2.VideoCapture(0)

ret, frame = cap.read()
background = np.zeros(frame.shape)
#frame = cv2.resize(frame, (100, 50))
#background = cv2.resize(background, (100, 50))

#fgbg = mog.MOG(frame)
fgbg = cv2.createBackgroundSubtractorMOG2()


while(1):
    ret, frame = cap.read()
#    frame = cv2.resize(frame, (100, 50)) 
    if (not ret):
        break
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',frame)
#    cv2.imshow('output',fgmask)
    
    bgmask = (fgmask<THRESHOLD)[:,:,None]
    fgmask = (fgmask>(128 - THRESHOLD))[:,:,None]
    new_bg = frame*bgmask
    new_fg = frame*fgmask
    bg_prob_mask = np.ones(frame.shape) - (bgmask*CONFIDENCE)
    background = np.asarray((bg_prob_mask*background + (1-bg_prob_mask)*new_bg),np.uint8)
    foreground = np.asarray(fgmask*new_fg,np.uint8)
    cv2.imshow('background',background)
    cv2.imshow('foreground',foreground)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


