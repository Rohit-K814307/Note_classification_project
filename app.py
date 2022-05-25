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
from scipy.io import wavfile
from scipy.fftpack import fft
from pylab import *
import streamlit as st

from funcs2 import *
from UploadPage import *

#_______________________________________________________________________________________

page = st.sidebar.selectbox("Navigate The Page!", "P")
refresh = st.button("Record again!")

if page == "P" or refresh == True:
    p_page()


