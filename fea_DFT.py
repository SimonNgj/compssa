# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:00:02 2019

@author: xngu0004
"""

import numpy as np
import random
import scipy.io as sio
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn import svm
from scipy.fftpack import fft

list_skill = ["Suture", "Knot", "Needle"]
list_win = [90, 120, 150, 180]
list_over = [30, 60]
list_netw = [1]

# users in Group
novice = ['B', 'G', 'H', 'I']
intermediate = ['C', 'F']
expert = ['D', 'E']

f_s = 30 # Sampling rate
denominator = 10

# Function to calculate FFT coefficiences
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(T), N)
    fft_values_ = fft(y_values)
    fft_values = 1.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

# Revised from http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, ax=None):

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind

# Extract no_peaks
def get_first_n_peaks(x,y,no_peaks=10):
    x_, y_ = list(x), list(y)
    if len(x_) >= no_peaks:
        return x_[:no_peaks], y_[:no_peaks]
    else:
        missing_no_peaks = no_peaks-len(x_)
        return x_ + [0]*missing_no_peaks, y_ + [0]*missing_no_peaks
    
def get_features(x_values, y_values, mph):
    indices_peaks = detect_peaks(y_values, mph=mph)
    peaks_x, peaks_y = get_first_n_peaks(x_values[indices_peaks], y_values[indices_peaks])
    return peaks_x + peaks_y

# Extract features
def extract_features_labels(dataset, labels, T, N, f_s, denominator):
    percentile = 5
    list_of_features = []
    list_of_labels = []
    for signal_no in range(0, len(dataset)):
        features = []
        list_of_labels.append(labels[signal_no])
        for signal_comp in range(0,dataset.shape[2]):
            signal = dataset[signal_no, :, signal_comp]
            
            signal_min = np.nanpercentile(signal, percentile)
            signal_max = np.nanpercentile(signal, 100-percentile)
            ##ijk = (100 - 2*percentile)/10
            mph = signal_min + (signal_max - signal_min)/denominator
            
            #features += get_features(*get_psd_values(signal, T, N, f_s), mph)
            features += get_features(*get_fft_values(signal, T, N, f_s), mph)
            #features += get_features(*get_autocorr_values(signal, T, N, f_s), mph)
        list_of_features.append(features)
    return np.array(list_of_features), np.array(list_of_labels)

for win in list_win:
    for over in list_over:
        for skill in list_skill:
            " Load the data"
            print('loading the dataset')
            dataset = sio.loadmat('../data_Mar8/data/'+skill+'_dataX_basic_'+str(win)+'_'+str(over)+'.mat')
            features = dataset['dataX']
            sizeD_ts = (((features[0,0:1])[0])[0,:])[0].shape[0]
            sizeD_va = (((features[0,0:1])[0])[0,:])[0].shape[1]
            N = sizeD_ts # Number of samples
            t_n = N/f_s # Duration of window
            T = 1 / f_s # Period
            print('loaded the dataset')
            for netw in list_netw:
                for tim in range(3,4):
                    acc_1 = []
                    accu = 0
                    acc_1_fea = []
                    accu_fea = 0                    
                    ###################################
                    #Pre-process data
                    for user in range(1,6):
                        " Initialize data for concatenation "
                        X_test1 = np.ones((1,sizeD_ts,sizeD_va))
                        y_test1 = np.ones(1)
                        X_train1 = np.ones((1,sizeD_ts,sizeD_va))
                        y_train1 = np.ones(1)
                        print("User out: ", user)
                        # Extract data for training and testing
                        for j in range(0,3): # 3 means 3 groups
                            for i in range(0, (((features[0,j:j+1])[0])[0,:]).shape[0]):
                                k = (((features[0,j:j+1])[0])[2,:])[i]
                                if (k == user): # take out the ith trial of each user for testing
                                    temp_x = (((features[0,j:j+1])[0])[0,:])[i]
                                    temp_x1 = temp_x.reshape(1,sizeD_ts,sizeD_va)
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
                                    temp_x1 = temp_x.reshape(1,sizeD_ts,sizeD_va)
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
                        Xx_test, yy_test = shuffle(X_test1, y_test1, random_state = random.randint(10,50))
                        Xx_train, yy_train = shuffle(X_train1, y_train1, random_state = random.randint(10,50))
            
                        ##############################################
                        X_train, Y_train = extract_features_labels(Xx_train, yy_train, T, N, f_s, denominator)
                        X_test, Y_test = extract_features_labels(Xx_test, yy_test, T, N, f_s, denominator)
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
                        f = open(str(tim)+'_'+skill+'_'+str(win)+'_'+str(over)+'_'+str(netw)+'_report_DFT_fea.txt', 'a+')
                        f.write("-------------------- DFT Features -----------------------")
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