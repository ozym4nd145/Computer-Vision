import cv2
import numpy as np
import os
import argparse

from LKT import lkt


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Video Stabilization')
    parser.add_argument('vid', metavar='V',
                        help='Input video to be stabilized')
    parser.add_argument('out', metavar='O',
                        help='path to the write the output video')
    parser.add_argument('--nowrite', action="store_true",
                        help='do not write the output video')
    parser.add_argument('--show',  action="store_true",
                        help='show the output video')
    args = parser.parse_args()

    video_file = args.vid
    outp_vid = args.out

    cols,rows = (200,300)
    DIVIDER_LEN=20

    cap = cv2.VideoCapture(video_file)
    ret,frame=  cap.read()
    resz_frame = cv2.resize(frame,(rows,cols))
    
    divider = np.ones([resz_frame.shape[0],DIVIDER_LEN,resz_frame.shape[2]],dtype=resz_frame.dtype)*255
    
    r = cv2.selectROI("Select window of interest",img=resz_frame,fromCenter=False,showCrossair=False)
    cv2.destroyAllWindows()

    tmpl = resz_frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    initial_warp = np.float32([[1.0,0.0,-float(r[0])],[0.0,1.0,-float(r[1])]])
    warp = initial_warp.copy()

    if not args.nowrite:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(5) ## get fps of video
        vid = cv2.VideoWriter(outp_vid, fourcc, fps, (rows*2+DIVIDER_LEN, cols))

    while True:
        ret,frame=  cap.read()
        if not ret:
            break
        resz_frame = cv2.resize(frame,(rows,cols))
        warp = lkt( resz_frame, tmpl, warp, 0.0010, transform='AFFINE')
        frame_warp = warp.copy()
        frame_warp[0,2] -= initial_warp[0,2]
        frame_warp[1,2] -= initial_warp[1,2]
        warped_img = cv2.warpAffine(resz_frame,frame_warp,(rows,cols))

        result = np.concatenate((resz_frame,divider,warped_img),axis=1)
    

        if not args.nowrite:
            vid.write(result)
        
        if args.show:
            # cv2.imshow("original",resz_frame)
            # cv2.imshow("result",warped_img)
            cv2.imshow("result",result)
            ret = cv2.waitKey(20)
            if (ret & 0xFF == ord('q'))or(ret ==27):
                break
    cv2.destroyAllWindows()
    if not args.nowrite:
        vid.release()