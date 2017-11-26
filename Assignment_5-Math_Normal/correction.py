import cv2
import numpy as np
import os
import argparse
import sys

def show_images(imgList):
    while True:
        for (name,img) in imgList:
            cv2.imshow(name,img)
        pressed_key = cv2.waitKey(0)
        if pressed_key==27:
            break
    cv2.destroyAllWindows()

def inhomogenize(p):
    return p/p[2]

def get_affine(p1,p2,p3,p4):
    ## p1 p2 p3 p4 in anticlockwise order of square
    p1,p2,p3,p4 = np.concatenate(([p1,p2,p3,p4],np.ones([4,1],dtype=np.float32)),axis=1)
    
    l1 = inhomogenize(np.cross(p1,p2))
    l2 = inhomogenize(np.cross(p3,p4))
    l3 = inhomogenize(np.cross(p2,p3))
    l4 = inhomogenize(np.cross(p1,p4))
    
    a = inhomogenize(np.cross(l1,l2))
    b = inhomogenize(np.cross(l3,l4))
    
    linf = inhomogenize(np.cross(a,b))
    
    return np.array([[1,0,0],[0,1,0],[linf[0],linf[1],1]],dtype=np.float32)


def get_metric(p1,p2,p3,p4):
    ## p1 p2 p3 p4 in anticlockwise order of square
    img_points = np.array([p1,p2,p3,p4],dtype=np.float32)
    size = (np.linalg.norm(p1-p2)
            +np.linalg.norm(p2-p3)
            +np.linalg.norm(p3-p4)
            +np.linalg.norm(p4-p1))/4
    cube_points = np.array([[p1[0],p1[1]],
                            [p1[0],p1[1]+size],
                            [p1[0]+size,p1[1]+size],
                            [p1[0]+size,p1[1]]],dtype=np.float32)
    return cv2.getPerspectiveTransform(img_points,cube_points)

def get_points(img):
    points = [];
    img_to_show = img.copy()
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img_to_show,(x,y),4,(255,0,0),-1)
            points.append([x,y])
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    while(1):
        cv2.imshow('image',img_to_show)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    return points

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='Image Correction')
    parser.add_argument('img', metavar='I',
                        help='Input image to be corrected')
    parser.add_argument('out', metavar='O',
                        help='path to the output folder')
    parser.add_argument('--factor', metavar='F', default=1.0, type=float,
                        help='Factor to scale up the output image')
    parser.add_argument('--nowrite', action="store_true",
                        help='do not write the output images')
    parser.add_argument('--show',  action="store_true",
                        help='show the output image')
    args = parser.parse_args()

    img_file = args.img
    output_folder = args.out
    scale = args.factor

    img_name,img_ext = os.path.splitext(os.path.basename(img_file))
    aff_file = os.path.join(output_folder,img_name+"_aff.jpg")
    met_file = os.path.join(output_folder,img_name+"_met.jpg")

    img = cv2.imread(img_file)

    points = np.array(get_points(img),dtype=np.float32)
    ## Points should be in order of top-left, bot-left, bot-right, top-right
    
    H_met = get_metric(*points)
    H_aff = get_affine(*points)

    met_img = cv2.warpPerspective(img,H_met,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    aff_img = cv2.warpPerspective(img,H_aff,(int(scale*img.shape[1]),int(scale*img.shape[0])))

    if not args.nowrite:
        ## ensure existence of base folder
        os.makedirs(output_folder,exist_ok=True)

        cv2.imwrite(aff_file,aff_img)
        cv2.imwrite(met_file,met_img)

    if args.show:
        show_images([("Orig",img),("Metric",met_img),("Affine",aff_img)])