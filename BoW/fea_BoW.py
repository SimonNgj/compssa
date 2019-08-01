# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 21:57:12 2019

@author: xngu0004
"""

import numpy as np
import random
import scipy.io as sio
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn import svm

list_skill = ["Suture","Knot","Needle"]
list_win = [90,120,150,180]
list_over = [30,60]

# users in Group
novice = ['B', 'G', 'H', 'I']
intermediate = ['C', 'F']
expert = ['D', 'E']

for skill in list_skill:
    for over in list_over:
        for win in list_win:
            for tim in range(1,6):
                acc_1 = []
                accu = 0
                acc_1_fea = []
                accu_fea = 0                    
                ###################################
                #Pre-process data
                for user in range(1,6):
                    " Load the data"
                    print('loading the dataset')
                    dataset = sio.loadmat('./data/feature/'+skill+str(win)+'_'+str(over)+'_FeatureHist_8_100_'+str(user)+'.mat')
                    features = dataset['feature_hist']
                    sizeD_ts = (((features[0,0:1])[0])[0,:])[0].shape[0]
                    sizeD_va = (((features[0,0:1])[0])[0,:])[0].shape[1]
                    print('loaded the dataset')
            
                    " Initialize data for concatenation "
                    X_test1 = np.ones((1,100))
                    y_test1 = np.ones(1)
                    X_train1 = np.ones((1,100))
                    y_train1 = np.ones(1)
                    print("User out: ", user)
                    # Extract data for training and testing
                    for j in range(0,3): # 3 means 3 groups
                        for i in range(0, (((features[0,j:j+1])[0])[0,:]).shape[0]):
                            k = (((features[0,j:j+1])[0])[2,:])[i]
                            if (k == user): # take out the ith trial of each user for testing
                                temp_x = (((features[0,j:j+1])[0])[0,:])[i]
                                temp_x1 = temp_x.reshape(1,100)
                                X_test1 = np.concatenate((X_test1, temp_x1), axis=0) 
                                user_temp = (((features[0,j:j+1])[0])[1,:])[i]
        
                                if(user_temp in novice):
                                    y_test1 = np.concatenate((y_test1,[0]))
                                elif(user_temp in intermediate):
                                    y_test1 = np.concatenate((y_test1,[1]))
                                elif(user_temp in expert):
                                    y_test1 = np.concatenate((y_test1,[2]))
                            else:
                                temp_x = (((features[0,j:j+1])[0])[0,:])[i]
                                temp_x1 = temp_x.reshape(1,100)
                                X_train1 = np.concatenate((X_train1, temp_x1), axis=0)
                                user_temp = (((features[0,j:j+1])[0])[1,:])[i]
        
                                if(user_temp in novice):
                                    y_train1 = np.concatenate((y_train1,[0]))
                                elif(user_temp in intermediate):
                                    y_train1 = np.concatenate((y_train1,[1]))
                                elif(user_temp in expert):
                                    y_train1 = np.concatenate((y_train1,[2]))
        
                    " Delete the first column "
                    X_test1 = np.delete(X_test1,np.s_[0:1], axis=0)
                    y_test1 = np.delete(y_test1,np.s_[0:1])
                    X_train1 = np.delete(X_train1,np.s_[0:1], axis=0)
                    y_train1= np.delete(y_train1,np.s_[0:1])
                        
                    print("Shuffle data")
                    " Shuffle the training and testing data "
                    X_test, Y_test = shuffle(X_test1, y_test1, random_state = random.randint(10,50))
                    X_train, Y_train = shuffle(X_train1, y_train1, random_state = random.randint(10,50))
            
                    ##############################################
                        
                    svm1 = svm.SVC(kernel = 'linear')
                    svm1.fit(X_train, Y_train)
                    prediction_fea = svm1.predict(X_test)
                    y_pred_fea = prediction_fea
                    cr_fea = classification_report(Y_test, y_pred_fea)
                    cm_fea = confusion_matrix(Y_test, y_pred_fea)
    
                    acc_fea = np.sum(y_pred_fea == Y_test)/y_pred_fea.shape[0]
                    acc_1_fea.append(acc_fea)
                    print('Accuracy: ' + str(acc_fea))
                    print(cr_fea)
                    print(cm_fea)
                    f = open(str(tim)+'_'+skill+'_'+str(win)+'_'+str(over)+'_report_BOW_fea2.txt', 'a+')
                    f.write("-------------------- BOW Features -----------------------")
                    f.write('---------' + skill + " Win: " + str(win) + " Over: " + str(over) + " Trial: " + str(user) + '---------\n\n')
                    f.write('Accuracy: ' + str(acc_fea))
                    f.write('\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\n'.format(cr_fea, cm_fea))
                for i in range(0,5):
                    accu_fea += acc_1_fea[i]
                accu_fea = accu_fea/5
                f.write("\n-------------------------------------------\n")
                f.write("\nAcc_features = " + str(accu_fea))
                f.write("\n-------------------------------------------\n")
                f.close()