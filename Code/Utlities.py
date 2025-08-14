import cv2
from sklearn.model_selection import train_test_split
import glob
from Enhance_image import Enhance_image
import os
import numpy as np
import matplotlib.pyplot as plt

def read_images(IMAGES_PATH):
    ##Reading all the images files from the its path 
    all_images = [img for img in IMAGES_PATH]
    all_images.sort()
    return all_images



def Image_label(image):
    ##take the all_images as its input
    label=os.path.basename(image)
    return label


def Return_class(image):
    return Image_label(image).split('_')[0]



def Processing_dataset(all_images,IMAGES_PATH):
    '''
    this function prepare the he data set for extractin the features and matcing
    1-> gets the label of each image 
    2->convert the image to gray scale 
    3->apply image Enhancement
    4->splitting the data into train and test

    

    '''
    train_set = {}
    test_set = {}
    data = []       ## this will be a lis of tuples to hold the image and its label
    temp_label = '101'  # set class 101
    
    for filename in IMAGES_PATH:
        image = cv2.imread(filename,0)
        img,bin=Enhance_image(image)
        label = Image_label(filename)
        print('Processing image {}  '.format(Return_class(label)))
        data.append((Return_class(label), img))

        '''
        if temp_label != Return_class(filename):
            train, test =train_test_split(data, test_size=.1, random_state=5)
            train_set.update(train)
            test_set.update(test)
            temp_label = Return_class(filename)
            data=[]
        
       
        
        if filename == all_images[len(all_images) - 1]:
            train, test =train_test_split(data, test_size=.1, random_state=5)
            train_set.update(train)
            test_set.update(test)
        '''
           

    print('Preprocessing Done !')
    return data
