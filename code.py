import mne
import pandas as pd
import h5py
from scipy.io import loadmat
import scipy.signal as sg
from scipy.integrate import simps
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os

def get_epochs(file_path, annot_filepath):
    data = loadmat(file_path)['record']
    annot = loadmat(annot_filepath)['anno_apnea']
    fs = data.shape[1]/22470 # will be different for each patient
    epoch_len = int(fs*60) # 1 min.
    epochs = np.array([data[:,x-epoch_len:x] for x in range(epoch_len,data.shape[1]+1,epoch_len)]);print(epochs.shape)
    epochs = np.reshape(epochs, (epoch_len,data.shape[0],data.shape[1]//epoch_len))
    targets = annot[:, :epochs.shape[-1]].flatten()
    normal_epochs = epochs[:,:,targets==0]
    apnea_epochs = epochs[:,:,targets==1]
    print(normal_epochs.shape, apnea_epochs.shape)
    return normal_epochs, apnea_epochs, targets

root_folder='../input/sleep-edf-and-apnea/10318976/Data/UCDPaper/'
patients=['ucddb002','ucddb003','ucddb012','ucddb013',
         'ucddb018','ucddb020','ucddb022','ucddb024',
         'ucddb026','ucddb027']
train, test=patients[:1], patients[1]
train_epochs_normal,test_epochs_normal=[],[]
train_epochs_apnea,test_epochs_apnea=[],[]
train_labels, test_labels=[],[]
for i in range(len(train)):
    normal_ep,apnea_ep,targ = get_epochs(root_folder+train[i]+'/'+train[i]+'.mat',
                                        root_folder+train[i]+'/'+train[i]+'_anno.mat')
    train_epochs_normal.append(normal_ep)
    train_epochs_apnea.append(apnea_ep)
    train_labels.append(targ)
train_epochs_normal

x=train_epochs_normal[0]
x2 = train_epochs_apnea[0]
x = np.reshape(x, (x.shape[0]*x.shape[2],x.shape[1]))
x=x[:40000]
x2 = np.reshape(x2, (x2.shape[0]*x2.shape[2], x.shape[1]))
print(x.shape,x2.shape)
X=np.vstack([x,x2])
X.shape
y = np.zeros(X.shape[0])
y[x.shape[0]:] = 1
print(y.sum())
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
X = X[:,:5]
x_shuffled,y_shuffled = shuffle(X,y)
x_shuffled_y_shuffled = MinMaxScaler().fit_transform(x_shuffled, y_shuffled)
n_split=int(1.*X.shape[0])
x_train,x_test,y_train,y_test = X[:n_split], X[n_split:], y[:n_split],y[n_split:]
print(x_train.shape,y_train.shape)
clf = GaussianNB().fit(x_train,y_train)
train_acc = clf.score(x_train,y_train)
print(train_acc)

pred = clf.predict(X[39000:50000])
fig=plt.figure(figsize=(18,5))
plt.plot(y[39000:50000])
plt.plot(pred)
plt.legend(['true','predicted'])

delta = 1,4
theta = 4,8
alpha = 8,13
beta = 13,30
gamma = 30,100

targets.shape

test_file = loadmat("../input/sleep-edf-and-apnea/10318976/Data/UCDPaper/ucddb002/ucddb002.mat")

data = test_file['record'];print(data.shape)

fs = data.shape[1]/22470
fs

info = mne.create_info(ch_names=['unknown'] * 14,
                       ch_types=['misc'] * 14,
                       sfreq=fs)
raw = mne.io.RawArray(data, info)
raw.plot(show_scrollbars=False, show_scalebars=False)

raw.plot_psd(area_mode='range', tmax=60*6, picks='misc')
# raw.notch_filter(np.arange(60, 241, 60), picks='misc')
raw.plot_psd(area_mode='range', tmax=360.0, picks=range(5))

epoch_len = 128*60 # 1 min.
epochs = np.array([data[:,x-epoch_len:x] for x in range(epoch_len,data.shape[1]+1,epoch_len)]);print(epochs.shape)
epochs = np.reshape(epochs, (epoch_len,data.shape[0],data.shape[1]//epoch_len))
data_epochs = mne.EpochsArray(epochs, info)
# data_epochs.plot(picks='misc')

"EEG (C3-A2), EEG (C4-A1), left EOG, right EOG, submental EMG, \
ECG (modified lead V2), oro-nasal airflow (thermistor), ribcage movements, abdomen movements (uncalibrated strain gauges),\
oxygen saturation (finger pulse oximeter), snoring (tracheal microphone) and body position."
fig,ax=plt.subplots(14, figsize=(18,42))
for i in range(14):
    ax[i].plot(data[i,int(1.5e6):])

psd=loadmat("../input/sleep-edf-and-apnea/10318976/Data/UCDPaper/ucddb002/ucddb002_RQA_PSD.mat");print(psd.keys())
annot=loadmat("../input/sleep-edf-and-apnea/10318976/Data/UCDPaper/ucddb002/ucddb002_anno.mat");print(annot.keys())

plt.plot(psd['RMSValue'])
fig=plt.figure()
plt.plot(psd['RQA_LVM'])
fig=plt.figure()
plt.plot(psd['ZeroRQA_LVM'])

fig=plt.figure();plt.plot(annot['anno'].T)
fig=plt.figure();plt.plot(annot['anno_hypo'].T)
fig=plt.figure();plt.plot(annot['anno_apnea'].T)
fig=plt.figure();plt.plot(annot['anno_apnea_non_obstructive'].T)

len(annot['anno_apnea'].T)

targets = annot['anno_apnea'][:, :epochs.shape[-1]].flatten()
epochs.shape, targets.shape

normal_epochs = epochs[:,:,targets==0]
apnea_epochs = epochs[:,:,targets==1]
normal_epochs.shape, apnea_epochs.shape

normal_epochs = epochs[:,:,targets==0]
apnea_epochs = epochs[:,:,targets==1]
normal_epochs.shape, apnea_epochs.shape

alphaband = betaband = thetaband = deltaband = gammaband = np.zeros(6)

for i in range(6):
    alphaband[i] = bandpower(apnea_epochs[:,0,i][600:], 128, alpha);
    betaband[i] = bandpower(apnea_epochs[:,0,i][600:], 128, beta);
    thetaband[i] = bandpower(apnea_epochs[:,0,i][600:], 128, theta);
    deltaband[i] = bandpower(apnea_epochs[:,0,i][600:], 128, delta);
    gammaband[i] = bandpower(apnea_epochs[:,0,i][600:], 128, gamma);
    
fig = plt.figure(); plt.scatter(range(len(alphaband)),alphaband);
fig = plt.figure(); plt.scatter(range(len(thetaband)),thetaband);

plt.plot(normal_epochs[:,0,0][600:])

spects_apnea = [sg.spectrogram(apnea_epochs[:,0,i][600:], fs=fs, nperseg=int(10*fs)) for i in range(len(apnea_epochs[0,0,:]))]
spects_normal = [sg.spectrogram(normal_epochs[:,0,i][600:], fs=fs, nperseg=int(10*fs)) for i in range(len(normal_epochs[0,0,:]))]

plt.plot(spects_apnea[5][0],spects_apnea[5][2])

spects = spects_apnea + spects_normal # 6 + 368 epochs
len(spects)
X = np.asarray(spects)

labels = len(apnea_epochs[0,0,:])*[1]+len(normal_epochs[0,0,:])*[0]
len(labels)
y = np.asarray(labels, dtype=float)