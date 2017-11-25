import cv2
import numpy as np

def show_images(imgList):
    while True:
        for (name,img) in imgList:
            cv2.imshow(name,img)
        pressed_key = cv2.waitKey(0)
        if pressed_key==27:
            break
    cv2.destroyAllWindows()

def jacob_warp_affine(x,y):
    return np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]],dtype=np.float32) ##(2,3)

def linearize_warp(wf):
    assert (wf.shape==(2,3))
    return np.float32([[wf[0,0]-1.0],[wf[1,0]],[wf[0,1]],[wf[1,1]-1.0],[wf[0,2]],[wf[1,2]]]) ##(6,1)

def make_warp(p):
    assert (p.shape==(6,1))
    return np.float32([[p[0,0]+1.0,p[2,0],p[4,0]],[p[1,0],p[3,0]+1.0,p[5,0]]]) ##(2,3)

def inverse_param(p):
    assert (p.shape==(6,1))
    denom = ((1+p[0,0])*(1+p[3,0]) - (p[1,0]*p[2,0]))
    invp = np.zeros(p.shape,dtype=np.float32)
    invp[0,0] = (-p[0,0])-(p[0,0]*p[3,0])+(p[1,0]*p[2,0])
    invp[1,0] = (-p[1,0])
    invp[2,0] = (-p[2,0])
    invp[3,0] = (-p[3,0])-(p[0,0]*p[3,0])+(p[1,0]*p[2,0])
    invp[4,0] = (-p[4,0])-(p[3,0]*p[4,0])+(p[2,0]*p[5,0])
    invp[5,0] = (-p[5,0])-(p[0,0]*p[5,0])+(p[1,0]*p[4,0])
    invp = invp/denom
    return invp ##(6,1)

def compose_param(p1,p2):
    assert (p1.shape==(6,1))
    assert (p2.shape==(6,1))
    comp = np.zeros(p1.shape,dtype=np.float32)
    comp[0,0] = p1[0,0]+p2[0,0]+(p1[0,0]*p2[0,0])+(p1[2,0]*p2[1,0])
    comp[1,0] = p1[1,0]+p2[1,0]+(p1[1,0]*p2[0,0])+(p1[3,0]*p2[1,0])
    comp[2,0] = p1[2,0]+p2[2,0]+(p1[0,0]*p2[2,0])+(p1[2,0]*p2[3,0])
    comp[3,0] = p1[3,0]+p2[3,0]+(p1[1,0]*p2[2,0])+(p1[3,0]*p2[3,0])
    comp[4,0] = p1[4,0]+p2[4,0]+(p1[0,0]*p2[4,0])+(p1[2,0]*p2[5,0])
    comp[5,0] = p1[5,0]+p2[5,0]+(p1[1,0]*p2[4,0])+(p1[3,0]*p2[5,0])
    return comp ##(6,1)

def gaussNewton(img,template,warp,epsilon,transform):
    template = template.astype('float32')
    img = img.astype('float32')
    assert (len(img.shape)==2), "Grayscale image should be given"
    ## template is the target image
    ## img is initial image
    ## warp is the initial guess warp ([1+p1,p3,p5],[p2,1+p4,p6])
    ## transform is one of "AFFINE" "TRANSLATIONAL"
    ## epsilon defines the ending condition
    

    template_shape = template.shape
    tmplx,tmply = template_shape
    
    ## Calculating gradients (3)
    
    # kerX = np.float32([[-0.5,0.0,0.5]])
    # kerY = np.float32([-0.5,0.0,0.5])
    # gradX = cv2.filter2D(template,-1,kerX)
    # gradY = cv2.filter2D(template,-1,kerY)
    
    gradTx = cv2.Sobel(template,cv2.CV_64F,1,0,ksize=3) ##(x,y)
    gradTy = cv2.Sobel(template,cv2.CV_64F,0,1,ksize=3) ##(x,y)
    grad = np.stack([gradTx,gradTy],axis=-1) ##(x,y,2)
    grad = np.expand_dims(grad,axis=2) ##(x,y,1,2)
    
    ## Calculating jacobian of template image (4)
    jacob_tmpl = np.zeros([tmplx,tmply,2,6]) ## (x,y,2,6)
    for i in range(tmplx):
        for j in range(tmply):
            # because i,j in numpy image correspond to j,i in image axis
            jacob_tmpl[i,j] = jacob_warp_affine(j,i)
    
    ## Calculating steepest descent (5)
    steep_desc = np.matmul(grad,jacob_tmpl) ##(x,y,1,6)
    steep_desc_trans = np.transpose(steep_desc,[0,1,3,2]) ##(x,y,6,1)
    
    
    ## Calculating Hessian matrix (6)
    hess = np.sum(np.sum(np.matmul(steep_desc_trans,steep_desc),axis=0),axis=0) ##(6,6)
    inv_hess = np.linalg.inv(hess) ##(6,6)
    
    delP = np.ones([6,1],dtype=np.float32)
    iterations = 0
#     while(np.linalg.norm(delP,2) > epsilon): #2-Norm end condition
    while(np.linalg.norm(delP) > epsilon): #Frobenius norm end condition
        
        ## Calculation warp of given image with current guess (1)
        warp_img = cv2.warpAffine(img,warp,(tmply,tmplx)) ##(x,y)
        ## Calculate error image (2)
        err_img = warp_img - template ##(x,y)
        err_img = np.expand_dims(np.expand_dims(err_img,-1),-1) ##(x,y,1,1)
        
        ## Computer other term (7)
        other_term = np.sum(np.sum(np.matmul(steep_desc_trans,err_img),axis=0),axis=0) ##(6,1)
        ## Computing delP (8)
        delP = np.matmul(inv_hess,other_term) ##(6,1)
        
        ## Updating warp (9)
        initP = linearize_warp(warp) ##(6,1)
#         invP = inverse_param(delP) ##(6,1)
        invP = delP
        nextP = compose_param(initP,invP) ##(6,1)
        warp = make_warp(nextP) ##(2,3)
        iterations += 1
        
#         print("Iteration %d ; Norm %f" %(iterations,np.linalg.norm(delP,ord='fro')))
        
        if iterations > 2500:
            break
    return warp
    
def lkt(img1,img2,initialwarp,epsilon, transform='AFFINE'):
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    return gaussNewton(img1,img2,initialwarp,epsilon,transform)

