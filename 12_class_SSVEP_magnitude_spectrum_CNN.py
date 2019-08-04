# -*- coding: utf-8 -*-
"""
12-Class SSVEP EEG Dataset
Classification Using Convolutional Neual Network

User-Dependent Training using Magnitude Spectrum Features (10-Fold Cross-validation)

Following implementation is an asynchronous SSVEP BCI using Convolutional Neural Network classification for 1 second data length

Citation/Reference:
Ravi, A., Manuel, J., Heydari, N., & Jiang, N. A Convolutional Neural Network for 
Enhancing the Detection of SSVEP in the Presence of Competing Stimuli. 
In IEEE Engineering in Medicine and Biology Conference 2019.

Implementation:
Aravind Ravi
eBionics Lab
University of Waterloo
"""
import scipy.io as sio
from scipy.signal import butter, filtfilt
import math
import numpy as np
from sklearn.model_selection import KFold
import keras 
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import optimizers, initializers, regularizers


#Buffer implementation similar to MATLAB - Segmentation of signal/windowing
def buffer(data,duration,data_overlap):
    number_segments = int(math.ceil((len(data)-data_overlap)/(duration-data_overlap)))
    #print(data.shape)
    temp_buf = [data[i:i+duration] for i in range(0,len(data),(duration-int(data_overlap)))]
    temp_buf[number_segments-1] = np.pad(temp_buf[number_segments-1],(0,duration-temp_buf[number_segments-1].shape[0]),'constant')
    temp_buf2 = np.vstack(temp_buf[0:number_segments])
    return temp_buf2

#Digital Filter Bandpass zero-phase Implementation (filtfilt)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data)
    return y

#Variable to store all accuracies
all_acc=np.zeros((10,1))

#Iterate through all subjects datasets
for sub in range(0,10):

    #Load EEG Dataset
    dataset=sio.loadmat('s'+str(sub+1)+'.mat')
    
    eeg=np.array(dataset['eeg'],dtype='float32')

#Reading the required parameters
#Number of classes	
    num_classes=eeg.shape[0]
    
#Number of EEG channels
    num_chan=eeg.shape[1]

#Trial length of EEG
    total_trial_len=eeg.shape[2]

#Total number of trials
    num_trials=eeg.shape[3]
    sample_rate = 256

#SSVEP flicker frequencies used for the 12 SSVEP targets   
    flicker_freq=np.array([9.25,11.25,13.25,9.75,11.75,13.75,10.25,12.25,14.25,10.75,12.75,14.75])
    
    labels=[]

#Extract the actual trial from the data (refer the paper for more details)
    trial_len=int(38+0.135*sample_rate+4*sample_rate-1) - int(38+0.135*sample_rate)
    filtered_data = np.zeros((eeg.shape[0],eeg.shape[1],trial_len,eeg.shape[3]))

#Iterate through the channels and trials for filtering the data using Bandpass filter between 6 Hz and 80 Hz
    for cl in range(0,num_classes):
        for ch in range(0,num_chan):
            for tr in range(0,num_trials):    
                filtered_data[cl,ch,:,tr] = butter_bandpass_filter(np.squeeze(eeg[cl,ch,int(38+0.135*sample_rate):int(38+0.135*sample_rate+4*sample_rate-1),tr]),6,80,sample_rate,4)
    
    eeg=[]

#Segment the EEG trials into 1 second non-overlapping segments 
#Window/segment length in seconds
    window_len=1

#Shift of the window in seconds (indirectly specifies overlap)
    shift_len=1

#Converting into units of samples
    duration=int(window_len*sample_rate)
    data_overlap = (window_len-shift_len)*sample_rate
    
    number_of_segments = int(math.ceil((filtered_data.shape[2]-data_overlap)/(duration-data_overlap)))
    
    segmented_data = np.zeros((filtered_data.shape[0],filtered_data.shape[1],filtered_data.shape[3],number_of_segments,duration))

#Performing data segmentation on each trial and each channel for all classes of data
    for cl in range(0,num_classes):
        for ch in range(0,num_chan):
            for tr in range(0,num_trials):
                segmented_data[cl,ch,tr,:,:]=buffer(filtered_data[cl,ch,:,tr],duration,data_overlap)
    
    filtered_data=[]
    freqRef=[]

#Computing the magnitude spectrum features for every epoch and extarcting the components between 3 Hz and 35 Hz
    fft_len = segmented_data[0,0,0,0,:].shape[0]

#FFT resolution
    resolution = 0.2930
    NFFT1 = round(sample_rate/resolution)
    start_freq = 3.0
    fft_index_start = int(round(start_freq/resolution))
    end_freq = 35.0
    fft_index_end = int(round(end_freq/resolution))+1

#Empty array to hold the data
    features_data = np.zeros((fft_index_end-fft_index_start,segmented_data.shape[1],segmented_data.shape[0],segmented_data.shape[2],segmented_data.shape[3]))
    
    labels = []

#Computing the magnitude spectrum features for every segment and all trials/epochs
    for cl in range(0,num_classes):
        for ch in range(0,num_chan):
            for tr in range(0,num_trials):
                for sg in range(0,number_of_segments):
                    temp_FFT = np.fft.fft(segmented_data[cl,ch,tr,sg,:],NFFT1)/fft_len
                    temp_mag=2*np.abs(temp_FFT)
                    features_data[:,ch,cl,tr,sg] = temp_mag[fft_index_start:fft_index_end,]
                                    
    segmented_data = []

#Combining the features into a matrix of dim [features X channels X classes X trials*segments]
    features_data = np.reshape(features_data,(features_data.shape[0],features_data.shape[1],features_data.shape[2],features_data.shape[3]*features_data.shape[4]))
    
    train_data=features_data[:,:,0,:].T

#Reshaping the data into dim [classes*trials*segments X channels X features]
    for cl in range(1,features_data.shape[2]):
        train_data = np.vstack([train_data,np.squeeze(features_data[:,:,cl,:]).T])

#Finally reshaping the data into dim [classes*trials*segments X channels X features X 1]    
    train_data = np.reshape(train_data,(train_data.shape[0],train_data.shape[1],train_data.shape[2],1))
    
    tot_class_epoch=features_data.shape[3]
    features_data=[]
    
#True labels in the training data
    labels=np.vstack([0*np.ones((tot_class_epoch,1)),1*np.ones((tot_class_epoch,1)),2*np.ones((tot_class_epoch,1)),3*np.ones((tot_class_epoch,1)),4*np.ones((tot_class_epoch,1)),5*np.ones((tot_class_epoch,1)),6*np.ones((tot_class_epoch,1)),7*np.ones((tot_class_epoch,1)),8*np.ones((tot_class_epoch,1)),9*np.ones((tot_class_epoch,1)),10*np.ones((tot_class_epoch,1)),11*np.ones((tot_class_epoch,1))])

#One-hot encoding of the labels
    labels = to_categorical(labels)

#10-fold crossvalidation - creating the folds 
    kf = KFold(n_splits=10,shuffle=True)

#fold counter
    fold=-1
    kf.get_n_splits(train_data)
    cv_acc=np.zeros((10,1))

#Train-test split and Convolutional neural network training
    for train_index, test_index in kf.split(train_data):
        x_tr, x_ts = train_data[train_index], train_data[test_index]
        y_tr, y_ts = labels[train_index], labels[test_index]
        fold=fold+1
        
#Training parameters
#Mini-Batch size
        batch_size = 64

#Number of Epochs for training
        epochs = 50 

#Dropout regularization - dropout ratio
        droprate=0.25

#Learning rate - alpha
        learning_rate=0.001

#Learning rate decay (if required)
        lr_decay=0.0
        
        print("Subject:",sub+1,"Fold:",fold+1,"Training...")

#Convolutional Neural Network Architecture
        model = Sequential()
#Spatial filtering across the EEG channels
        model.add(Conv2D(2*num_chan, kernel_size=(num_chan,1), input_shape=(x_tr.shape[1],x_tr.shape[2],x_tr.shape[3]),padding="valid",kernel_regularizer=regularizers.l2(0.0001),kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(droprate))  
#Feature extarction across spectral domain
        model.add(Conv2D(2*num_chan, kernel_size=(1, 10), kernel_regularizer=regularizers.l2(0.0001),padding="valid",kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(droprate))  
        model.add(Flatten())
#Classification layer
        model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l2(0.0001),kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
        
        #print(model.summary())
        
#Model optimizers - Stochastic Gradient Descent with Momentum        
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=False)

#Compile the model - with loss function as categorical cross entropy
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd,metrics=["accuracy"])
        
        history = model.fit(x_tr, y_tr,batch_size=batch_size,epochs=epochs,verbose=0)

#Evaluate the model on the test data
        score = model.evaluate(x_ts, y_ts,verbose=0) 
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#Accumulate the accuracy for each fold
        cv_acc[fold,:]=score[1]*100
        
#Mean accuracy across the 10-folds
    all_acc[sub] = np.mean(cv_acc)
    print("...................................................")
    print("Subject:",sub+1, " - Accuracy:",all_acc[sub],"%")
    print("...................................................")
    trainData=[]

#Overall accuracy across all subjects
print(".....................................................................................")
print("Overall Accuracy Across Subjects:",np.mean(all_acc),"%","std:",np.std(all_acc),"%")
print(".....................................................................................")
