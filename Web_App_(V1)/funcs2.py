import librosa
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment
import wave
import matplotlib.pyplot as plt
from os import path
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
import tensorflow.keras.backend as K
from PIL import Image
import cv2
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import wavfile # get the api
from scipy.fftpack import fft
from pylab import *
import streamlit as st

#__________________________________________________________

def scale(X):
  scaler = StandardScaler()

  X = scaler.fit_transform(X)

  return X

#________________________________________________________________

def plotSaveSound(x,filename):
  wav_obj = wave.open(x, 'rb')


  #important values
  sample_freq = wav_obj.getframerate()
  n_samples = wav_obj.getnframes()
  time = n_samples/sample_freq
  n_channels = wav_obj.getnchannels()
  #print(filename)


  #wave over time for frequency calculations
  signal_wave = wav_obj.readframes(n_samples)
  signal_array = np.frombuffer(signal_wave, dtype=np.int16)

  l_channel = signal_array[0::2]
  r_channel = signal_array[1::2]

  #plotting
  plt.figure(figsize=(3.6, 3.6))
  plt.specgram(l_channel, Fs=sample_freq, vmin=-20, vmax=50)
  #plt.title(filename)
  #plt.ylabel('Frequency (Hz)')
  #plt.xlabel('Time (s)')
  plt.axis('off')
  plt.xlim([2.5,6])

  plt.savefig('C:\\Users\\kulka\\NoteClasf_webapp\\imsnew\\' + str(filename))

  #plt.show()

  #__________________________________________

def find_target(X):
  le = preprocessing.LabelEncoder()
  target2 = le.fit_transform(X)
  return target2

def createSoundAndTarget(directory,newArr=[]):
  for filen in glob.iglob(f'{directory}/*'):
    dog = str(filen[80:])
    newArr.append(dog[:-1])
    plotSaveSound(filen)
    y_data = np.array(newArr)
    target = find_target(y_data)
    return target

def createData(dir,X_data=[]):
  imgdirectory = "C:\\Users\\kulka\\NoteClasf_webapp\\"+str(dir)
  for myfile in glob.iglob(f'{imgdirectory}/*'):
    image = cv2.imread(myfile)
    X_data.append(image)
  X_data2 = np.array(X_data)
  return X_data2

def createData2(dir,X_data=[]):
  im = Image.open(str(dir))
  st.image(im, caption='Wave of the audio you recorded!')
  image = cv2.imread(dir)
  X_data.append(image)
  X_data2 = np.array(X_data)
  X_data3 = np.asarray(X_data2).astype('float32')
  st.write(str(X_data3.shape))
  return X_data3

#______________________________________________

def newPred(audio,model):

  plotSaveSound(audio)

  data = createData('imsnew')
  

  y_pred = model.predict(data)

  num_to_note = {0:"3G",9:"6E",10:"6F",11:"6F#",12:"6G",5:"5A",7:"5Bb",6:"5B",8:"5C",1:"4D",3:"4Eb",2:"4E",4:"4F"}

  out = np.argmax(y_pred)
  fin = num_to_note.get(out)
  print(fin)

  #____________________________________________________

def mod():

  X_data = createData("imgconv2")
  target = np.array([ 0,  9,  9, 10, 10, 11, 11, 12, 12,  5,  5,  7,  7,  6,  6,  8,  8, 1,  1,  3,  3,  2,  2,  4,  4,  0])
  X_train, X_test, y_train, y_test = train_test_split(X_data, target, test_size=0.1, random_state=42,shuffle=True)

  model=Sequential()
  model.add(Conv2D(filters=1,kernel_size=2,activation="relu",input_shape=(360,360,3)))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Conv2D(filters=2,kernel_size=2,activation ="relu"))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Conv2D(filters=4,kernel_size=2,activation ="relu"))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Conv2D(filters=8,kernel_size=2,activation ="relu"))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Conv2D(filters=16,kernel_size=2,activation ="relu"))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Flatten())
  model.add(Dense(13,activation="relu"))
  model.add(Dense(13,activation="softmax"))
  

  optimizer='adam'
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  metrics=['accuracy']

  model.compile(optimizer=optimizer,loss = loss, metrics=metrics)
  model.fit(X_train,y_train)

  return model