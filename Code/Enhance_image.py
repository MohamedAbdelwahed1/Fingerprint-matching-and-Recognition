import numpy as np
import cv2
import math
import scipy
from scipy import signal
from scipy import ndimage
from Image_Preprocessing import *


### Enhancing image by preprocessing 
def Enhance_image(img):

    ##parameter intilaizatin##
    ridge_blocksize=16
    Ridge_threshold=.1
    sigma_gradient=1
    sigma_block=7
    sigma_smooth=5
    block_size=38
    block_window=5
    
    minWlength=5
    MaxWlength=15

    Renormaized_image ,Mask= Ridge_Segementaion(img,ridge_blocksize,Ridge_threshold )
    image_orientian=Ridge_orientain(Renormaized_image,sigma_gradient,sigma_block,sigma_smooth)
    #print( image_orientian)
    Ridge_frequencie=Ridge_frequencies(Renormaized_image,image_orientian,Mask,
    block_size,block_window,minWlength,MaxWlength)
    #print( Ridge_frequencie)
    binary_image,newim =gaborfilter(Renormaized_image,Ridge_frequencie,image_orientian) 


    return newim,binary_image
























