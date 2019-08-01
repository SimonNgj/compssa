# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:50:37 2019

@author: xngu0004
PCA on python
"""

import numpy as np
import random
import scipy.io as sio
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#####################################################
"Select a configuration"

list_skill = ["Suture", "Knot", "Needle"]
# users in Group
novice = ['B', 'G', 'H', 'I']
intermediate = ['C', 'F']
expert = ['D', 'E']
        
win_list = [90,120,150,180]
over_list = [30,60]

####################################################
for win in win_list:
    for over in over_list:
        for skill in list_skill:
            " Load the data"
            dataset = sio.loadmat(skill+'_dataX_prePCA_'+str(win)+'_'+str(over)+'.mat')
            features = dataset['dataA']
            for tim in range(1,6):
                acc_1_fea = []
                accu_fea = 0  
                ###############################

                print('Pre-process data')
                for trial in range(1,6):

                    " Initialize data for concatenation "
                    X_test1 = np.ones((1,features.shape[1]-2))
                    y_test1 = np.ones(1)
                    X_train1 = np.ones((1,features.shape[1]-2))
                    y_train1 = np.ones(1)

                    # Extract data for training and testing
                    for i in range(0, features.shape[0]):
                        k = features[i,1]
                        if (k == trial): # take out the ith trial of each user for testing
                            temp_x = features[i,2:]
                            temp_x1 = temp_x.reshape(1,features.shape[1]-2)
                            X_test1 = np.concatenate((X_test1, temp_x1), axis=0) 
                            user = int(features[i,0])
        
                            if(chr(user) in novice):
                                y_test1 = np.concatenate((y_test1,[0]))
                            elif(chr(user) in intermediate):
                                y_test1 = np.concatenate((y_test1,[1]))
                            elif(chr(user) in expert):
                                y_test1 = np.concatenate((y_test1,[2]))
                        else:
                            temp_x = features[i,2:]
                            temp_x1 = temp_x.reshape(1,features.shape[1]-2)
                            X_train1 = np.concatenate((X_train1, temp_x1), axis=0)
                            user = int(features[i,0])
        
                            if(chr(user) in novice):
                                y_train1 = np.concatenate((y_train1,[0]))
                            elif(chr(user) in intermediate):
                                y_train1 = np.concatenate((y_train1,[1]))
                            elif(chr(user) in expert):
                                y_train1 = np.concatenate((y_train1,[2]))
        
                    " Delete the first column "
                    X_test1 = np.delete(X_test1,np.s_[0:1], axis=0)
                    y_test1 = np.delete(y_test1,np.s_[0:1])
                    X_train1 = np.delete(X_train1,np.s_[0:1], axis=0)
                    y_train1 = np.delete(y_train1,np.s_[0:1])

                    " Shuffle the training and testing data "
                    X_test, y_test = shuffle(X_test1, y_test1, random_state = random.randint(10,50))
                    X_train, y_train = shuffle(X_train1, y_train1, random_state = random.randint(10,50))
                    #################################################### 
                    sc = StandardScaler()  
                    X_train = sc.fit_transform(X_train)  
                    X_test = sc.transform(X_test)
                    
                    pca = PCA(n_components=50)  
                    X_train = pca.fit_transform(X_train)  
                    X_test = pca.transform(X_test) 
                    ####################################################      
                    print("Skill: ", skill)
                    print("Trial out:", trial)
                    f = open('./results/'+str(tim)+'_'+skill+'_'+str(win)+'_'+str(over)+'_report_prePCA_fea2_50.txt', 'a+')
            
                    svm1 = svm.SVC(kernel = 'linear')
                    svm1.fit(X_train, y_train)
                    prediction_fea = svm1.predict(X_test)
                    y_pred_fea = prediction_fea
                    cr_fea = classification_report(y_test, y_pred_fea)
                    cm_fea = confusion_matrix(y_test, y_pred_fea)
    
                    acc_fea = np.sum(y_pred_fea == y_test)/y_pred_fea.shape[0]
                    acc_1_fea.append(acc_fea)
                    print('Accuracy: ' + str(acc_fea))
                    print(cr_fea)
                    print(cm_fea)
                    f.write('---------' + skill + " Win: " + str(win) + " Shift: " + str(over) + " Trial: " + str(trial) + '---------\n\n')
                    f.write('Accuracy: ' + str(acc_fea))
                    f.write('\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n\n'.format(cr_fea, cm_fea))
                for i in range(0,5):
                    accu_fea += acc_1_fea[i]
                accu_fea = accu_fea/5
                f.write("\n-------------------------------------------\n")
                f.write("Acc_fea = " + str(accu_fea))
                f.write("\n-------------------------------------------\n")
                f.close()