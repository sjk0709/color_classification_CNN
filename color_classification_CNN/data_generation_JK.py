#!/usr/bin/env python

import sys, os
sys.path.append(os.pardir)  # parent directory
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
#from sklearn.feature_extraction import image

from PIL import Image
import cv2

import glob
import random
import struct


from skimage import io, color

# PIL_JK class includes PIL util made by JK

class Data(object):
    def __init__(self):
        self.images = np.zeros(1)
        self.labels = np.zeros(1)
        self.start_batch = 0
        self.end_batch = 0
        self.num_examples = 0
        
    def next_batch(self, batch_size):
        mini_batch = np.random.choice(len(self.images), batch_size, replace=False)
        
#        self.end_batch = self.start_batch+batch_size
#        mini_batch = np.arange(self.start_batch, self.end_batch)
#        if self.end_batch!=len(self.images):
#            self.start_batch = self.end_batch
#        else :
#            self.start_batch = 0
        
        return self.images[mini_batch], self.labels[mini_batch]
              
           
def rotateImg(img, angle):    
    rows = img.shape[0]
    cols = img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    image = cv2.warpAffine(img,M,(cols,rows))
    return image
 
   
import ntpath
ntpath.basename("a/b/c")

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


import ImageProcessing_JK


def im2double(img):
    info = np.iinfo(img.dtype)  # Get the data type of the input image
    return img.astype(np.float) / info.max  # Divide all values by the largest possible value in the data type


def correct_FlatImg(img, kernel_size):

    # -- 1. Convert color image: BGR -> RGB
#    fileImg_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
    
    # -- 2. Split color image
    # b, g, r = cv2.split(fileImg_BGR)       # get b, g, r
    # fileImg_RGB = cv2.merge([r, g, b])     # merge to rgb

    r, g, b = cv2.split(img)  # get r, g, b

    # cv2.imwrite('test1_r.bmp', r)
    # cv2.imwrite('test1_g.bmp', g)
    # cv2.imwrite('test1_b.bmp', b)

    # -- 3. Convert to floating point: not normalized (use numpy)
    # out = cv2.normalize(fileImg_gray.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    floatImg_r = im2double(r)
    floatImg_g = im2double(g)
    floatImg_b = im2double(b)

    # -- 4. R, G, B average
    means_r = cv2.mean(floatImg_r)
    means_g = cv2.mean(floatImg_g)
    means_b = cv2.mean(floatImg_b)
    # print('means_r: ', means_r)
    # print('means_g: ', means_g)
    # print('means_b: ', means_b)

    # -- 5. Gaussian
    floatImg_r_blur = cv2.GaussianBlur(floatImg_r, (kernel_size, kernel_size), 0)
    floatImg_g_blur = cv2.GaussianBlur(floatImg_g, (kernel_size, kernel_size), 0)
    floatImg_b_blur = cv2.GaussianBlur(floatImg_b, (kernel_size, kernel_size), 0)

    # -- 6. 
    diffImg_r = floatImg_r - floatImg_r_blur + means_r[0]
    diffImg_g = floatImg_g - floatImg_g_blur + means_g[0]
    diffImg_b = floatImg_b - floatImg_b_blur + means_b[0]

    # -- 7. : float -> uint8 (8 bit)
    fileImg_out_r = 255 * diffImg_r
    fileImg_out_g = 255 * diffImg_g
    fileImg_out_b = 255 * diffImg_b

    # cv2.imwrite('test2_r.bmp', fileImg_out_r)
    # cv2.imwrite('test2_g.bmp', fileImg_out_g)
    # cv2.imwrite('test2_b.bmp', fileImg_out_b)

    # -- 8. 
    flatImg_BGR = cv2.merge([fileImg_out_b, fileImg_out_g, fileImg_out_r])  # merge to bgr -> save
    flatImg_BGR = np.array(flatImg_BGR, dtype=np.uint8)

    # -- 9.
    # cv2.imwrite('testImg_BGR.bmp', flatImg_BGR)

    return flatImg_BGR

def convertRGB2Lab(imageList):    
    for imgNo in range(len(imageList)):  
        tempImg = imageList[imgNo].copy()               
        imageList[imgNo]  = color.rgb2lab(tempImg)  
#                self._sampleArr[imgNo] = cv2.cvtColor(self._sampleArr[imgNo], cv2.COLOR_RGB2Lab) 
    return imageList 

def getSamples(path, width, height, color_type='RGB', feature_type='full'): # input : path  # output : imgList   # path안의 이미지들을 리스트로 만들어준다.  
   
    imgPro = ImageProcessing_JK.ImageProcessing_JK( outputFolderPath='data_set')
   
    imageList = []
#    fileNameList = []
    labelList = []      
      
    for filepath in glob.glob(path+'/*.bmp'):    # make a image list with images in path
        
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)  # B, G, R   
        
        # crop and enhance image
        x_start = 380
        x_end = 1460
        y_start = 310
        y_end = 750        
        croppedImg = image[y_start:y_end, x_start:x_end]    # Crop a image  
        sampleImg = imgPro.findSampleImage(croppedImg, width=width, height=height )  
          
        if(color_type =='RGB'):
            sampleImg = cv2.cvtColor(sampleImg, cv2.COLOR_BGR2RGB)     # BGR -> RGB
        elif(color_type =='Lab'):
            sampleImg = cv2.cvtColor(sampleImg, cv2.COLOR_BGR2Lab)     # BGR -> Lab
        elif(color_type=='Gray'):
            sampleImg = cv2.cvtColor(sampleImg, cv2.COLOR_BGR2GRAY)     # BGR -> Gray
        
#        cv2.imshow('image', sampleImg)
#        cv2.waitKey(0)
        
        if (feature_type=='full'):
            tempImg = sampleImg
        else:
            if(color_type =='BGR'):
                sampleImg = cv2.cvtColor(sampleImg, cv2.COLOR_BGR2RGB)     # BGR -> RGB
            elif(color_type =='Lab'):
                sampleImg = cv2.cvtColor(sampleImg, cv2.COLOR_Lab2RGB)     # BGR -> Lab
            elif(color_type =='Gray'):
                sampleImg = cv2.cvtColor(sampleImg, cv2.COLOR_GRAY2RGB)     # BGR -> Lab
            kernel_size = 5   
            tempImg = correct_FlatImg(sampleImg, kernel_size=kernel_size)
            
        
        fileName = path_leaf(filepath)       
        
        label = 'Null'
        if (fileName.find('ref') != -1):
            label='REF'
        elif (fileName.find('OK') != -1):
            label='OK'
        elif (fileName.find('NG') != -1):
            label='NG'
            
        imageList.append(tempImg)
#        fileNameList.append(fileName)
        labelList.append(label) 

    return imageList, labelList

class Galaxy(object):
    """
    """
    def __init__(self, folder_path, sampleSize, feature_shape, color_type='Lab', feature_type='full', normalization_type='each_images', 
                 nExtract=6000, extraction_type='random', withScatter=True, nFakes=5):
       
        self._path = folder_path
        self._sampleSize = sampleSize    
        self._feature_shape = feature_shape
        self._color_type = color_type
        self._feature_type = feature_type
        self._normalization_type = normalization_type
        self._withScatter = withScatter
        self._maxScatter = 25000
        self._nFakes = nFakes
                  
        self._trainImages, self._trainImageLabels = getSamples( self._path + '/train', self._sampleSize[0], self._sampleSize[1],color_type=self._color_type, feature_type=self._feature_type)
        self._testImages, self._testImageLabels = getSamples( self._path + '/test', self._sampleSize[0], self._sampleSize[1], color_type=self._color_type, feature_type=self._feature_type)
       
        self._nTrainImages = len(self._trainImages)
        self._nTestImages = len(self._testImages)
        self._nImages = self._nTrainImages + self._nTestImages
        self._nChanels = self._feature_shape[2]        
        
        self._imageWidth = self._sampleSize[0]      
        self._imageHeight = self._sampleSize[1]
  
        self._blockW = feature_shape[0]
        self._blockH = feature_shape[1]

        self._nExtract = nExtract
        
        #===== convert RGB to L-ab or Gray =====
        if (self._color_type=='Galaxy'):
            print('Read Galaxy data')
        elif (self._color_type=='Lab' and self._withScatter==False):
            print('Read Galaxy L-ab data without scatter')
        elif (self._color_type=='Lab' and self._withScatter==True):
            self._nChanels = 4
            print('Read Galaxy L-ab data with scatter')
        elif (self._color_type=='Gray'):
            self._nChanels = 1
            print('Read Galaxy Gray data')
        else:
            print('Read Galaxy RGB data')
            
        #self._save_path = "D:\\JKfactory\\my_projects\\ML\\projects\\DAGM_vae_tf\\DAGM"
        self._save_path = folder_path
        self._load_path = self._save_path
        if not os.path.exists(self._save_path):
            os.mkdir(self._save_path)

        #readFreeImg()
        self.train = Data()
        self.test = Data()                  
        
        #===== convert RGB to Lab =====#
#        if(self._color_type=='Lab'):
#            self._trainImages = convertRGB2Lab(self._trainImages)
#            self._testImages = convertRGB2Lab(self._testImages)

        # normalize images
        self._images_mean = 0
        self._images_std = 0        
#        self._normalizeImages()
            
        self.genImgData()
    
    def getMeanStd(self):
        return self._images_mean, self._images_std
        
    def _normalizeImages(self):
              
        if (self._normalization_type=='all_images'):
            self._trainImages = np.array(self._trainImages)
            self._images_mean = np.mean(self._trainImages, axis=0)
            self._images_std = np.std(self._trainImages, axis=0)
            
#            self._trainImages = (self._trainImages - self._images_mean) #/ self._images_std
#            self._testImages = (self._testImages - self._images_mean) #/ self._images_std
            
            print('images_mean shape : ', self._images_mean.shape)
            print('images_std shape : ', self._images_std.shape)
            
        elif (self._normalization_type=='each_images'):
            self._trainImages = np.array(self._trainImages)
            self._images_mean = np.mean(self._trainImages, axis=0)
            self._images_std = np.std(self._trainImages, axis=0)
            
            self._trainImages = (self._trainImages - self._images_mean) #/ self._images_std
            self._testImages = (self._testImages - self._images_mean) #/ self._images_std
            
            print('images_mean shape : ', self._images_mean.shape)
            print('images_std shape : ', self._images_std.shape)
            
        elif (self._normalization_type=='each_images_smoothing'):
            self._trainImages = np.array(self._trainImages)
            self._images_mean = np.mean(self._trainImages, axis=0)
            self._images_std = np.std(self._trainImages, axis=0)
            
            self._trainImages = (self._trainImages - self._images_mean) #/ self._images_std
            self._testImages = (self._testImages - self._images_mean) #/ self._images_std
            
            print('images_mean shape : ', self._images_mean.shape)
            print('images_std shape : ', self._images_std.shape)

 #       cv2.imshow('rgb2', self._images[0])
#        cv2.imshow('rgb3', self._images_mean)
#        cv2.waitKey(0)                 # Waits forever for user to press any key
#        cv2.destroyAllWindows()        # Closes displayed windows
#        
#        plt.figure(1)
#        plt.imshow(fileImg)
#        plt.figure(2)        
#        plt.imshow(self._images_mean, 'gray')
#        plt.colorbar()
#        plt.show()
#        
 
    
    
    def dataAugmentation(self, fileImg, start_x, end_x, start_y, end_y, Y):    
#        print( start_x, end_x, start_y, end_y)
        tmpXli = []
        tmpYli = []

        # -- crop a block
        img1 = fileImg[start_y:end_y, start_x:end_x]        
        img2 = cv2.flip(img1,1)
        img3 = rotateImg(img1, 180)
        img4 = rotateImg(img2, 180)
        
        tmpXli.append(img1)
        tmpYli.append(Y)      
        tmpXli.append(img2)
        tmpYli.append(Y)        
        tmpXli.append(img3)
        tmpYli.append(Y)        
        tmpXli.append(img4)
        tmpYli.append(Y)
        
#        if (self._normalization_type=='all_images' or self._normalization_type=='each_images_smoothing'):
#            img5 = rotateImg(img1, 90)            
#            img6 = rotateImg(img1, 270)
#            img7 = rotateImg(img2, 90)
#            img8 = rotateImg(img2, 270)
#            tmpXli.append(img5)
#            tmpYli.append(Y)            
#            tmpXli.append(img6)
#            tmpYli.append(Y)            
#            tmpXli.append(img7)
#            tmpYli.append(Y)            
#            tmpXli.append(img8)
#            tmpYli.append(Y)
        
#        cv2.imshow("vertical flip", rimg)
#        fimg=img1.copy()        
#        fimg=cv2.flip(img1,0)
#        cv2.imshow("horizontal flip", fimg)
#         wait time in milliseconds
#         this is required to show the image
#         0 = wait indefinitely
#        cv2.waitKey(0)
#         close the windows
#        cv2.destroyAllWindows()
#        
        
#        cv2.imshow("data", img1)
#        cv2.waitKey(0)
     
        return tmpXli, tmpYli
        

    
    def extractImgData(self, fileImg, label):
        
        Xli = []
        Yli = []
            
        if(self._feature_type=='full'):
            Xli.append(fileImg)           
            Yli.append(label)            
            Xli.append(rotateImg(fileImg, 180))
            Yli.append(label)
            
        else:            
            marginLeft = 100
            marginRight = 10
            marginUp = 100
            marginDown = 100
            maxFeatureWidth = self._imageWidth - marginLeft - marginRight
            maxFeatureHeight = self._imageHeight - marginUp - marginDown
            
            if (maxFeatureWidth<self._blockW) or (maxFeatureHeight<self._blockH):
                return print("Exception Error : feature size is too large")            
            
            xLimit = self._imageWidth - marginRight - self._blockW 
            yLimit = self._imageHeight - marginDown - self._blockH
    #        print('xNoLimit : ', xNoLimit, 'yNoLimit : ', yNoLimit)
            
            
                    
            if (self._normalization_type=='all_images' or self._normalization_type=='each_images_smoothing'):
                for i in range(self._nExtract):
                    start_x = random.randrange(marginLeft, xLimit)
                    end_x = start_x + self._blockW               
                    start_y = random.randrange(marginUp, yLimit)
                    end_y = start_y + self._blockH
                    
                    bufferX ,bufferY = self.dataAugmentation(fileImg, start_x, end_x, start_y, end_y, label)
                    Xli += bufferX
                    Yli += bufferY
                    
            elif (self._normalization_type=='each_images'):
                for i in range(self._nExtract):
                    start_x = random.randrange(2, xLimit)
                    end_x = start_x + self._blockW               
                    start_y = random.randrange(2, yLimit)
                    end_y = start_y + self._blockH
                 
                    bufferX ,bufferY = self.dataAugmentation(fileImg, start_x, end_x, start_y, end_y, label)
                    Xli += bufferX
                    Yli += bufferY
                
  
        
#        for j in range(2):
#            start_y_up = 15 + 10*j
#            start_y_center = 115 + 10*j
#            start_y_down = 215 + 10*j
#            for i in range(60):
#                start_x = 15 + 12*i
#                
#                end_x = start_x + self._blockW               
#
#                bufferX ,bufferY = self.dataAugmentation(fileImg, start_x, start_y_up, end_x, start_y_up+self._blockH, label)
#                Xli += bufferX
#                Yli += bufferY
#                bufferX ,bufferY = self.dataAugmentation(fileImg, start_x, start_y_center, end_x, start_y_center+self._blockH, label)
#                Xli += bufferX
#                Yli += bufferY
#                bufferX ,bufferY = self.dataAugmentation(fileImg, start_x, start_y_down, end_x, start_y_down+self._blockH, label)
#                Xli += bufferX
#                Yli += bufferY                
#        
#            
        return Xli, Yli
    
    
    def genImgData(self):

        trainX = 0
        testX = 0
        trainXList = []
        trainYList = []
        testXList = []
        testYList = []               

        if(self._nFakes>0):
            OKTrainList = []
            for imgNo in range(self._nTrainImages):
                
                fileImg = self._trainImages[imgNo]  
                label = self._trainImageLabels[imgNo]
    
                if label=='OK' or label=='REF': 
                    OKTrainList.append(fileImg)
    
    
            #####################################################################################################
               
            def genFakeImageByAvg(imageList, savePath=None):
                images_mean = np.mean(imageList, axis=0)    
    #            images_mean = images_mean.astype("uint8")
                if (savePath):
                    path = savePath + '.bmp'
    #                cv2.imwrite(path, images_mean)
                return images_mean
            
            nOKTrainList = len(OKTrainList)
            OKTrainList = np.array(OKTrainList)            
            for i in range(self._nFakes):
                indices = np.arange(nOKTrainList) # Get A Test Batch        
                np.random.shuffle(indices)
                nForAvg = random.randrange(2,nOKTrainList+1)
                indices = indices[0:nForAvg]
                temp = OKTrainList[indices]
                
                savePathFileName = 'OK_fake' + str(i) + '.bmp'
                ref_fake = genFakeImageByAvg(temp, savePathFileName) 
                
#                cv2.imshow('test', ref_fake/255)
#                cv2.waitKey(0)
                self._trainImages.append(ref_fake) 
                self._trainImageLabels.append('OK')
              
            OKTrainList = []
         #####################################################################################################       
                
     
     
        for imgNo in range(self._nTrainImages):
            
            fileImg = self._trainImages[imgNo]  
            label = self._trainImageLabels[imgNo]
            
            # label            
            Y = np.zeros([2], dtype='float')   # [1 0] : OK  |  [0 1] : NG
            
            if label=='OK' or label=='REF':           
                Y[0] = 1.
            elif label=='NG':                
                Y[1] = 1.   
                 
        
            crop_img_list, label_list = self.extractImgData(fileImg, Y)   
                   
            crop_img_arr = np.array(crop_img_list) /255.
            
            ## testing scatter
            if (self._withScatter):
                scatter = np.full([crop_img_arr.shape[0], crop_img_arr.shape[1], crop_img_arr.shape[2], 1], 15000/self._maxScatter)
                temp = np.concatenate((crop_img_arr, scatter), axis=3)
            else:
                temp = crop_img_arr
   
            if imgNo==0:
                trainX = temp
            else :
                trainX = np.concatenate((trainX, temp), axis=0)            
#            trainXList += crop_img_list
            trainYList += label_list
        
        print("===== Generating training data finished =====")
        
        for imgNo in range(self._nTestImages):
            
            fileImg = self._testImages[imgNo]  
            label = self._testImageLabels[imgNo]
            
            # label            
            Y = np.zeros([2], dtype='float')   # [1 0] : OK  |  [0 1] : NG
            
            if label=='OK' or label=='REF':           
                Y[0] = 1.
            elif label=='NG':                
                Y[1] = 1.
    
            crop_img_list, label_list = self.extractImgData(fileImg, Y)   
                    
            crop_img_arr = np.array(crop_img_list) /255.     
            ## testing scatter
            if (self._withScatter):
                scatter = np.full([crop_img_arr.shape[0], crop_img_arr.shape[1], crop_img_arr.shape[2], 1], 15000/self._maxScatter)
                temp = np.concatenate((crop_img_arr, scatter), axis=3)
            else:
                temp = crop_img_arr
            
            if imgNo==0:
                testX = temp
            else :
                testX = np.concatenate((testX, temp), axis=0)            
#            testXList += crop_img_list
            testYList += label_list
            
            
        print("===== Generating test data finished =====")

        self.train.images = trainX#.reshape(-1, self._feature_shape[0]*self._feature_shape[1]*self._feature_shape[2])
        self.train.labels = np.array( trainYList )
        self.test.images = testX#.reshape(-1, self._feature_shape[0]*self._feature_shape[1]*self._feature_shape[2])
        self.test.labels = np.array( testYList )
        
        self.train.num_examples = self.train.images.shape[0]
        self.test.num_examples = self.test.images.shape[0]
        
        print('train.images.shape : ', self.train.images.shape)
        print('train.labels.shape : ', self.train.labels.shape)        
        print('test.images.shape : ', self.test.images.shape)
        print('test.labels.shape : ', self.test.labels.shape)

        self.train.images = trainX.reshape(-1, self._feature_shape[0]*self._feature_shape[1]*self._feature_shape[2])        
        self.test.images = testX.reshape(-1, self._feature_shape[0]*self._feature_shape[1]*self._feature_shape[2])
        
        trainXList = []
        trainYList = []
        testXList = []
        testYList = []    
         
    
if __name__ == '__main__':  
    

    dataPath = '../../../color_defect_galaxy/data/train_data'
     
    color_type = 'Gray'  # RGB /  BGR  /  Lab
    feature_type = 'full'   # full  or  block
    nExtract = 50
    nFakes = 0
    withScatter = False
    
    sample_size = [1000, 300]    
    feature_shape = [32, 32, 1]   
    
    if(feature_type == 'full'):
        sample_size[0] = feature_shape[0]
        sample_size[1] = feature_shape[1]
        
    galaxy = Galaxy (dataPath, sampleSize=sample_size, feature_shape=feature_shape, color_type=color_type, feature_type=feature_type, nExtract=nExtract, 
                     withScatter=withScatter, nFakes=nFakes)
    
    batch_size = 5
        
    for i in range(1):
        trainX, trainY = galaxy.train.next_batch(batch_size)
        print('trainX.shape : ',trainX.shape)
        print(trainX)
        print('trainY.shape : ',trainY.shape)
        print(trainY)
        print('-----------------------------------------------------')

    
    images = galaxy.train.images
    labels = galaxy.train.labels
    print('images.shape : ', images.shape)
    print('labels.shape : ', labels.shape)

    cv2.imshow('test', images[0].reshape( feature_shape[1], feature_shape[0]))
    cv2.waitKey(0)