# -*- coding: utf-8 -*-
"""
SSVEP Dataset
Classification Using Canonical Correaltion Analysis (CCA)
The following is implemented on a 12-Class publicly available SSVEP EEG Dataset
Dataset URL: https://github.com/mnakanishi/12JFPM_SSVEP/tree/master/data
Paper: Masaki Nakanishi, Yijun Wang, Yu-Te Wang and Tzyy-Ping Jung, 
"A Comparison Study of Canonical Correlation Analysis Based Methods for Detecting Steady-State Visual Evoked Potentials," 
PLoS One, vol.10, no.10, e140703, 2015. 

Following implementation is an asynchronous SSVEP BCI using CCA classification for 1 second data length

Implementation:
Aravind Ravi
eBionics Lab
University of Waterloo
"""
import scipy.io as sio
from scipy.signal import butter, lfilter
from scipy.signal import freqz
from sklearn.cross_decomposition import CCA
from sklearn.metrics import confusion_matrix
import numpy as np
import math

#Buffer implementation similar to MATLAB - Segmentation of signal/windowing
def buffer(data,duration,dataOverlap):
    numberOfSegments = int(math.ceil((len(data)-dataOverlap)/(duration-dataOverlap)))
    #print(data.shape)
    tempBuf = [data[i:i+duration] for i in range(0,len(data),(duration-int(dataOverlap)))]
    tempBuf[numberOfSegments-1] = np.pad(tempBuf[numberOfSegments-1],(0,duration-tempBuf[numberOfSegments-1].shape[0]),'constant')
    tempBuf2 = np.vstack(tempBuf[0:numberOfSegments])
    return tempBuf2

#Digital Filter Implementation
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#Generate reference signals for canonical correlation analysis (CCA) for SSVEP detection
def getReferenceSignals(length, target_freq,samplingRate):
    # generate sinusoidal reference templates for CCA for the first and second harmonics
    reference_signals = []
    t = np.arange(0, (length/(samplingRate)), step=1.0/(samplingRate))
    #First harmonics/Fundamental freqeuncy
    reference_signals.append(np.sin(np.pi*2*target_freq*t))
    reference_signals.append(np.cos(np.pi*2*target_freq*t))
    #Second harmonics
    reference_signals.append(np.sin(np.pi*4*target_freq*t))
    reference_signals.append(np.cos(np.pi*4*target_freq*t))
    reference_signals = np.array(reference_signals)
    return reference_signals

#Perform Canonical Correaltion Analysis (CCA)
def findCorr(n_components,signal_data,freq):
    # Perform Canonical correlation analysis (CCA)
    # signal_data - consists of the EEG
    # freq - set of sinusoidal reference templates corresponding to the flicker frequency
    cca = CCA(n_components)
    corr=np.zeros(n_components)
    result=np.zeros(freq.shape[0])
    for freqIdx in range(0,freq.shape[0]):
        cca.fit(signal_data.T,np.squeeze(freq[freqIdx,:,:]).T)
        r_a,r_b = cca.transform(signal_data.T, np.squeeze(freq[freqIdx,:,:]).T)
        indVal=0
        for indVal in range(0,n_components):
            corr[indVal] = np.corrcoef(r_a[:,indVal],r_b[:,indVal])[0,1]
            result[freqIdx] = np.max(corr)
    return result
#Variable to store all accuracies
all_acc=np.zeros((10,1))

#Iterate through all 10 subjects 
for sub in range(0,10):

#Load the EEG Dataset
    dataset=sio.loadmat('s'+str(sub+1)+'.mat')
   
    eeg=np.array(dataset['eeg'],dtype='float32')

#Reading the required parameters
#Number of classes	
    num_classes=eeg.shape[0]
#Number of EEG channels
    num_chan=eeg.shape[1]
#Trial length of EEG
    trial_len=eeg.shape[2]
#Total number of trials
    num_trials=eeg.shape[3]
    sample_rate = 256
#SSVEP flicker frequencies used for the 12 SSVEP targets   
	flickFreq=np.array([9.25,11.25,13.25,9.75,11.75,13.75,10.25,12.25,14.25,10.75,12.75,14.75])
#variable to store the true labels    
	labels=[]
    filtered_data = np.zeros(eeg.shape)
#Iterate through the channels and trials for filtering the data using Bandpass filter between 6 Hz and 80 Hz
    for cl in range(0,num_classes):
        for ch in range(0,num_chan):
            for tr in range(0,num_trials):
                filtered_data[cl,ch,:,tr] = butter_bandpass_filter(eeg[cl,ch,:,tr],6,80,sample_rate,order=4)

#Extract the actual trial from the data (refer the paper for more details)
    filtered_data = filtered_data[:,:,int(38+0.135*sample_rate):int(38+0.135*sample_rate+4*sample_rate-1),:]
    eeg=[]
#Segment the EEG trials into 1 second non-overlapping segments 
#Window/segment length in seconds
	windowLen=1
#Shift of the window in seconds (indirectly specifies overlap)
    shiftLen=1

#Converting into units of samples
    duration=int(windowLen*sample_rate)
    data_overlap = (windowLen-shiftLen)*sample_rate
    
    numberOfSegments = int(math.ceil((filtered_data.shape[2]-data_overlap)/(duration-data_overlap)))
    
    segmented_data = np.zeros((filtered_data.shape[0],filtered_data.shape[1],numberOfSegments,duration,filtered_data.shape[3]))

#Performing data segmentation on each trial and each channel for all classes of data
    for cl in range(0,num_classes):
        for ch in range(0,num_chan):
            for tr in range(0,num_trials):
                segmented_data[cl,ch,:,:,tr]=buffer(filtered_data[cl,ch,:,tr],duration,data_overlap)
    
    filtered_data=[]
    freqRef=[]
#Generating the required sinusoidal templates for the given 12-class SSVEP classification
    for fr in range(0,len(flickFreq)):
        freqRef.append(getReferenceSignals(duration,flickFreq[fr],sample_rate))
    
    freqRef=np.array(freqRef,dtype='float32')
    
    predictedClass=[]
#Performing segment wise classification using CCA 
    for cl in range(0,num_classes):
        for tr in range(0,num_trials):
            for sg in range(0,numberOfSegments):
#True labels are created here
                labels.append(cl)
#Perform CCA on a single segment
                result=findCorr(1,segmented_data[cl,:,sg,:,tr],freqRef)
#Pick the class that corresponds to the maximum canonical correlation coefficient
                predictedClass.append(np.argmax(result)+1)
    
    segmented_data=[]
    labels=np.array(labels)+1
    predictedClass=np.array(predictedClass)
#creating a confusion matrix of true versus predicted classification labels
    cMat = confusion_matrix(labels, predictedClass)
#computing the accuracy from the confusion matrix
    accuracy=np.divide(np.trace(cMat),np.sum(np.sum(cMat)))
    all_acc[sub] = accuracy
    print("Subject:",sub+1, " - Accuracy:",accuracy*100,"%")
#Mean overall accuracy across all subjects
print("Overall Accuracy Across Subjects:",np.mean(all_acc)*100,"%","std:",np.std(all_acc)*100,"%")