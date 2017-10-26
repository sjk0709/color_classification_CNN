# USAGE
# python auto_canny.py --images images


import sys, os

# import the necessary packages
import numpy as np
import glob
import cv2
from skimage import io, color
import math
import scipy.spatial.distance



class ImageProcessing_JK:
    
    def __init__(self, camera='3CCD', outputFolderPath='data_set'):  
        
        self._outputFolderPath = outputFolderPath
        if not os.path.exists(self._outputFolderPath):
            os.mkdir(self._outputFolderPath)
            os.mkdir(self._outputFolderPath+'/Ref')
            os.mkdir(self._outputFolderPath+'/OK')
            os.mkdir(self._outputFolderPath+'/NG')
        
        self._camera=camera
    

    def autoCanny(self, image, sigma=0.33):
    	# compute the median of the single channel pixel intensities
    	v = np.median(image)
    
    	# apply automatic Canny edge detection using the computed median
    	lower = int(max(0, (1.0 - sigma) * v))
    	upper = int(min(255, (1.0 + sigma) * v))
    	edged = cv2.Canny(image, lower, upper)
    
    	# return the edged image
    	return edged
    
    #-------------------------------------------------------------------------------------------------------
    def enhanceImage(self, image, vertices, width=1000, height=300):            
#        (rows,cols,_) = image.shape
#        
#        #image center
#        u0 = (cols)/2.0
#        v0 = (rows)/2.0
#        
#        
#        #widths and heights of the projected image
#        w1 = scipy.spatial.distance.euclidean(vertices[0],vertices[1])
#        w2 = scipy.spatial.distance.euclidean(vertices[2],vertices[3])
#        
#        h1 = scipy.spatial.distance.euclidean(vertices[0],vertices[2])
#        h2 = scipy.spatial.distance.euclidean(vertices[1],vertices[3])
#        
#        w = max(w1,w2)
#        h = max(h1,h2)
#        
#        #visible aspect ratio
#        ar_vis = float(w)/float(h)
#        
#        #make numpy arrays and append 1 for linear algebra
#        m1 = np.array((vertices[0][0],vertices[0][1],1)).astype('float32')
#        m2 = np.array((vertices[1][0],vertices[1][1],1)).astype('float32')
#        m3 = np.array((vertices[2][0],vertices[2][1],1)).astype('float32')
#        m4 = np.array((vertices[3][0],vertices[3][1],1)).astype('float32')
#        
#        #calculate the focal disrance
#        k2 = np.dot(np.cross(m1,m4),m3) / np.dot(np.cross(m2,m4),m3)
#        k3 = np.dot(np.cross(m1,m4),m2) / np.dot(np.cross(m3,m4),m2)
#        print('k2 : ', k2)
#        print('k3 : ', k3)
#        
#        n2 = k2 * m2 - m1
#        n3 = k3 * m3 - m1
#        print('n2 : ', n2)
#        print('n3 : ', n3)
#        
#        n21 = n2[0]
#        n22 = n2[1]
#        n23 = n2[2]
#        
#        n31 = n3[0]
#        n32 = n3[1]
#        n33 = n3[2]
#        
#        s = 1.0
#        s_sq = s*s
#        f_sq = (-1.0/(n23*n33*s_sq)) * ( (n21*n31 - (n21*n33 + n23*n31)*u0 + n23*n33*u0*u0)*s_sq + (n22*n32 - (n22*n33+n23*n32)*v0 + n23*n33*v0*v0) )
#        print('f_sq : ', f_sq)
#        f_sq = np.abs(f_sq)  # if f_sq < 0
#        f = np.sqrt(f_sq)
#        
#        A = np.array([[f,0,u0],[0,s*f,v0],[0,0,1]]).astype('float32')
#        
#        At = np.transpose(A)
#        Ati = np.linalg.inv(At)
#        Ai = np.linalg.inv(A)
#        
#        #calculate the real aspect ratio
#        ar_real_sq = 0
#        if k2==1 or k3==1:
#            ar_real_sq = (n21*n21+n22*n22) / (n31*n31+n32*n32)
#        else:
#            ar_real_sq = np.dot(np.dot(np.dot(n2,Ati),Ai),n2)/np.dot(np.dot(np.dot(n3,Ati),Ai),n3)    
#            
#        ar_real = np.sqrt(ar_real_sq)
#        
#        if ar_real < ar_vis:
#            width = int(width)
#            height = int(width / ar_real)
#        else:
#            height = int(h)
#            width = int(ar_real * height)
        
        #W = int(w)
        #H = int(h)
        
        pts1 = np.array(vertices).astype('float32')
        pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
        
        #project the image with the new w/h
        M = cv2.getPerspectiveTransform(pts1,pts2)        
        resultImg = cv2.warpPerspective(image, M, (width, height))
#        cv2.imshow('result', resultImg)
#        cv2.waitKey(0)
            
        return resultImg
        #-------------------------------------------------------------------------------------------------------
        
    
    def findSampleImage(self, image, width=1000, height=300):
        
        
#        cv2.imshow("Cropped image", croppedImg)
        
        # Convert the image to grayscale, and blur it slightly
        grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurredImg = cv2.GaussianBlur( grayImg, (3, 3), 0)

        # apply Canny edge detection using a wide threshold, tight
        # threshold, and automatically determined threshold
        if(self._camera=='basler'):     
            cannyImg = cv2.Canny( blurredImg, 30, 35)  # 200   95 100
        elif(self._camera=='3CCD'):     
            cannyImg = cv2.Canny( blurredImg, 95, 100)  # 200   95 100    
#        cannyImg = self.autoCanny( blurredImg )
                
#        print('Image shape : ', image.shape)
#        print('Cropped image shape : ', croppedImg.shape)
#        print('cannyImg.shape', cannyImg.shape)
#        #print(cannyImg)           
#        cv2.imshow("Original", image)
#        cv2.imshow("Cropped image", croppedImg)
#        cv2.imshow("Edges", cannyImg)
#        cv2.waitKey(0)
        

        leftTop = False
        rightTop = False
        leftBottom = False            
        rightBottom = False
                        
        lastIndex_X = cannyImg.shape[1]-1
        lastIndex_Y = cannyImg.shape[0]-1            
        vertices = []     
        LT = (0,0)
        RT= (0,0)
        LB = (0,0)
        RB = (0,0)
               
        if(self._camera=='basler'):                
            for x in range(int(cannyImg.shape[1]/2)):
                if (cannyImg[0][x]>0 and leftTop==False):
                    LT = (x,0)
                    leftTop=True                    
                if (cannyImg[0][lastIndex_X-x]>0 and rightTop==False):
                    RT = (lastIndex_X-x,0)
                    rightTop=True   
                if (cannyImg[-1][x]>0 and leftBottom==False):
                    LB = (x, lastIndex_Y)
                    leftBottom=True                    
                if (cannyImg[-1][lastIndex_X-x]>0 and rightBottom==False):
                    RB = (lastIndex_X-x, lastIndex_Y)
                    rightBottom=True  
            
            vertices.append(LT)
            vertices.append(RT)
            vertices.append(LB)
            vertices.append(RB)
            #        print('vertices : ', vertices)
            # enhance the image
            resultImg = self.enhanceImage( image, vertices=vertices, width=width, height=height )
                      
        elif(self._camera=='3CCD'):     
            
            for y in range(int(cannyImg.shape[0]/2)):
                
                if (cannyImg[y][0]>0 and leftTop==False):
                    LT = (0,y)
                    leftTop = True
                if (cannyImg[y][-1]>0 and rightTop==False):   
                    RT = (cannyImg.shape[1]-1, y)
                    rightTop = True
                if (cannyImg[lastIndex_Y-y][0]>0 and leftBottom==False):
                    LB = (0,lastIndex_Y-y)
                    leftBottom = True
                if (cannyImg[lastIndex_Y-y][-1]>0 and rightBottom==False):   
                    RB = (lastIndex_X, lastIndex_Y-y)
                    rightBottom = True
    
            vertices.append(LT)
            vertices.append(RT)
            vertices.append(LB)
            vertices.append(RB)            
            #        print('vertices : ', vertices)            
            # enhance the image
            resultImg = self.enhanceImage( image, vertices=vertices, width=width, height=height )


            
        resultPath = self._outputFolderPath + '/' + path 
#        cv2.imwrite(resultPath, resultImg)
#        cv2.waitKey(0)
                
        lines = cv2.HoughLinesP( cannyImg, 1 ,np.pi/180, 100, minLineLength=1, maxLineGap=1)
        #print(lines)
        
        # Show and save images                
#        cv2.imshow("Original", image)
#        cv2.imshow("Cropped image", croppedImg)
#        cv2.imshow("Edges", cannyImg)
#        cv2.imshow("Result", resultImg)
#        cv2.imwrite("Cropped_image.bmp", croppedImg)  
#        cv2.imwrite('Edges.bmp', cannyImg)  

        cv2.waitKey(0)
                
        #for x1,y1,x2,y2 in lines:
        #    for index, (x3,y3,x4,y4) in enumerate(lines):
        #
        #        if y1==y2 and y3==y4: # Horizontal Lines
        #            diff = abs(y1-y3)
        #        elif x1==x2 and x3==x4: # Vertical Lines
        #            diff = abs(x1-x3)
        #        else:
        #            diff = 0
        #
        #        if diff < 10 and diff is not 0:
        #            del lines[index]
        #
        #gridsize = (len(lines) - 2) / 2

        return resultImg
   
        
#==================================================================================================================================    
img =[]
path = 'S8_A_02NG_01_L190.bmp'
image = cv2.imread(path) 
img.append(image)
img.append(image)
#print(image)

## list -> numpy.array
#img = np.array(img)
#for i in range(2):
#    img[i] = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)  # BGR -> RGB   
#    img[i] = color.rgb2lab(img[i])                # BGR -> Lab
#    img[i] = color.lab2rgb(img[i])                # BGR -> Lab
#    img[i] = img[i].astype(np.float32)
#    img[i] = cv2.cvtColor(img[i], cv2.COLOR_RGB2BGR)  # BGR -> RGB   
#    cv2.imshow('img', img[i])
#    cv2.waitKey(0)
#    
    
    



#imgPro = ImageProcessing_JK(camera='3CCD')  # 'basler'  or   '3CCD'
#
#print(path.find('OK'))
#end = path.rfind('_01')
#print ((path[end-2:end]))
#end = path.rfind('/')
#print ((path[end+1:-4]))
#
#if (path.find('OK') != -1):
#    print('gogogo')

##
##result = imgPro.findSampleImage(path)
#path = 'S8_A_add2/S8_A_add2_OK01_01_L190.bmp'
#start = path.find('OK')
#print (int(path[start+2:start+4]))
#print(path[:path.rfind('_')])
#image = cv2.imread(path)         
##result = imgPro.findImageData(image)
#print('result_shape', image.shape)
#print(int('0001'))

# load the image
#print('Image shape : ', image.shape)   
#cv2.imshow("Original", image)
#cv2.waitKey(0)

#import matplotlib.pyplot as plt
#
#x = [1, 2, 3, 4 ,5]
#y = [1, 4, 9, 6, 10]
#
#fig, ax = plt.subplots()
#
## instanciate a figure and ax object
## annotate is a method that belongs to axes
#ax.plot(x, y, 'ro',markersize=23)
#
### controls the extent of the plot.
#offset = 1.0 
#ax.set_xlim(min(x)-offset, max(x)+ offset)
#ax.set_ylim(min(y)-offset, max(y)+ offset)
#
## loop through each x,y pair
#for i,j in zip(x,y):
#    corr = -0.05 # adds a little correction to put annotation in marker's centrum
#    ax.annotate( str(j),  xy=(i + corr, j + corr))
#
#plt.show()