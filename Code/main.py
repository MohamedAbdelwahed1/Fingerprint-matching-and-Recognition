import cv2
import numpy as np
import operator
from matplotlib import pyplot as plt
from sympy import Reals
from Utlities import*
from Feature_extraction import Extract_features
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path

from sklearn.linear_model import LogisticRegression

'''
first Reading the image 
'''

##Train path  
path1='./'+'DB2_B'+'/*.tif'
train_images = glob.glob(path1)
print("images flders",train_images)
all_images1=read_images(train_images)

##Test path
'''
path2='./'+'test'+'/*.tif'
test_images = glob.glob(path2)
all_images=read_images(test_images)
all_images2=read_images(test_images)
'''

'''
preparing the trainning datase (cleaning , enhancment , splitting)
'''
print("Preprocessing Trainning Set....")
Train_data=Processing_dataset(all_images1,train_images)
#print("Preprocessing Test Set....")
#Test_data=Processing_dataset(all_images2,test_images)

'''
some info abou the dataset
'''



#cv2.waitKey(0)



#print(enchanced_img)
'''
fig, ax = plt.subplots(1, 2)
ax1, ax2 = ax.ravel()
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Original image')
ax1.axis('off')
ax2.imshow(train['101_1.tif'], cmap=plt.cm.gray)
ax2.set_title('Enhanced  image')
ax2.axis('off')
plt.show()
'''


##
##Perform Feature Extraction to apply SVM
##

Extracted_train_features,labels1=Extract_features(Train_data,500)
#Extracted_test_features,labels2=Extract_features(Test_data,500)

X_train, X_test, y_train, y_test = train_test_split(Extracted_train_features,labels1,test_size=.2)

#X_train, X_test, y_train, y_test = Extracted_train_features,Extracted_test_features,labels1,labels2
clf=SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

clf.fit(X_train, y_train)
preds = clf.predict(X_test)
training_score = clf.score(X_train, y_train)
acc = accuracy_score(y_test, preds)
test_score = clf.score(X_test, y_test)
print(training_score*100)
print(test_score*100)

conf_mat = confusion_matrix(y_test, preds)
print("Acc" ,acc*100)
print(conf_mat)
print("y_test",y_test)
'''
#### KNN  ####
clf = KNeighborsClassifier(n_neighbors = 2)
clf.fit(X_train, y_train)

training_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print(training_score*100)
print(test_score*100)


clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

training_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print(training_score*100)
print(test_score*100)

'''





###accuracy 62% ,clsters =100 ,c=1000
