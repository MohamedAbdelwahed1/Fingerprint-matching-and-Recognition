import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def Sift_descriptor(img, threshold):
    ORB= cv2.SIFT_create()    
    kp, descriptors = ORB.detectAndCompute(img.astype(np.uint8), None)
    return descriptors


def bag_of_features(features, centres, k ):
    '''
    inputs : 
    Features -> Sift descriptor 
    centres -> Kmeans Clustering centres

    Ouput :
    features_vector -> each feature with the ecuildien distance from the cluster
    '''

    features_vector = np.zeros((1, k))
    for i in range(features.shape[0]):
        feat = features[i]
        diff = np.tile(feat, (k, 1)) - centres
        dist = np.sqrt((pow(diff, 2)).sum(axis = 1))
        idx_dist = dist.argsort()
        idx = idx_dist[0]
        features_vector[0][idx] += 1
    return features_vector


def Extract_features (data,threshold) :
    features=[]
    for i in range (len(data)):
        img=data[i][1]
        label=data[i][0]
        img_des = Sift_descriptor(img,threshold)
        if img_des is not None:
         features.append(img_des)
     ##  features array of all images
    features = np.vstack(features)
    
    ## applying kmeans to get custers centres
    print(features.shape)
    num_clusters =180
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    #print("int8 image ")
    #plt.imshow(img.astype(np.uint8),cmap=plt.cm.gray)
    #plt.show()
    compactness, labels, centres = cv2.kmeans(np.float32(features), num_clusters, None, criteria, 10, flags)
    
    labels = []
    Extracted_features = []
    for i in range (len(data)):
        img=data[i][1]
        label=data[i][0]
        img_des = Sift_descriptor(img,threshold)
        img_vec = bag_of_features(img_des, centres,num_clusters)
        Extracted_features.append(img_vec)
        print(label)
        labels.append(label)
    Extracted_features = np.vstack(Extracted_features)

    return Extracted_features  , labels



###c=   
###orb= 
