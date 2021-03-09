#!/usr/bin/env python
# coding: utf-8

# <center><font size=3> Automatic License/Number Plate Recognition (ANPR) 

# In[1]:


from skimage.segmentation import clear_border
import numpy as np
import pytesseract
import imutils
import cv2


# In[85]:


class ANPR:
    def __init__(self, minAR=4, maxAR=5, debug=False):
        self.minAR = minAR # minimum aspect ratio used to detect and filter rectangular license plates
        self.maxAR = maxAR
        self.debug = debug #should display intermediate results in our image processing pipeline
        
    
    
    def debug_show(self, title, image, waitkey=False):
        if self.debug:
            cv2.imshow(title, image)
            
            if waitkey: # for single case 
                cv2.waitKey(0)
    
    #keep: num of numberplate stored license plate candidate contours
    # blackhat morphological operation to reveal dark characters
    #(letters, digits, and symbols) against light backgrounds
    def locate_license_plate_candidates(self, grayImg, keep=5):
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT,(13, 5)) # rect shape(13x5), typical international license plate shape.
        blackhat= cv2.morphologyEx(grayImg, cv2.MORPH_BLACKHAT, rectKern)
        self.debug_show("Black_Hat", blackhat)
        
        #region of light part images
        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        light = cv2.morphologyEx(grayImg, cv2.MORPH_CLOSE, squareKern)
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_show("light_region", light)
        
        #he Scharr gradient will detect edges in the image and emphasize the boundaries of the characters in the license plate:
        gradX = cv2.Sobel(blackhat, ddepth = cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal)= (np.min(gradX), np.max(gradX))
        gradX = 255 *((gradX- minVal)/ (maxVal- minVal))
        gradX = gradX.astype("uint8")
        self.debug_show("Scgarr", gradX)
        
        #smooth the group regions
        gradX = cv2.GaussianBlur(gradX, (5,5), 0)
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gradX, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        self.debug_show("Grad Thresh", gradX)
        
        #clear other parts of white region
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.debug_show("Grade erode/Dilate", thresh)
        
        # still need to be clear, put our light regions
        thresh = cv2.bitwise_and(thresh, thresh, mask= light)
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)
        self.debug_show("Final", thresh, waitkey=True)
        
        #sort contours
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:keep]
        
        return contours
    
    
    def locate_license_plate(self, grayImg, candidates, clearBorder = False):
        contours = None
        roi = None
        
        for c in candidates:
            (x, y, w, h) = cv2.boundingRect(c)
            ar =  w/ float(h)
            
            #if aspectRatio is rectangular
            if ar>=self.minAR and ar<=self.maxAR:
                contours = c
                licensePlate = grayImg[y:y+h, x:x+w]
                #binary-inverse thresholded using Otsu’s method
                roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                
                
                #IF WE HAVE CLEAR FORGROUND, 
                if clearBorder:
                    roi = clear_border(roi)
                
                self.debug_show("License Plate", licensePlate)
                self.debug_show("ROI, ", roi, waitkey= True)
                break
                
        # return a 2-tuple of the license plate ROI and the contour
        return (roi, contours)
    
    #psm => Tesseract Page Segmentation Mode
    def build_tesseract_options(self, psm=7):
        # tell Tesseract to only OCR alphanumeric characters
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        # set the PSM mode
        options += " --psm {}".format(psm)
        # return the built options string
        return options
    
    def find_and_ocr(self, image, psm=7, clearBorder = False):
        lpText = None
     
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.locate_license_plate_candidates(gray)
        (lp, lpCtn) = self.locate_license_plate(gray, candidates, clearBorder=clearBorder)
        
        if lp is not None:
            #OCR =>plate
            options = self.build_tesseract_options(psm=psm)
            lpText =  pytesseract.image_to_string(lp, config=options)
            self.debug_show("licence_Plate: ", lp)
            
        return (lpText, lpCtn)
    
    




