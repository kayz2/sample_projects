import os
import numpy as np
import pandas as pd
import importlib
import sys
import scipy.stats as st
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
#add CAN_bus_data_analysis folder to path so that files from this folder can
#be imported
sys.path.insert(0, 'CAN_bus_data_analysis')
import matplotlib.pyplot as plt

#Neural  network packages
import tensorflow as tf
from tensorflow import keras
from __future__ import print_function
from keras.layers import Dense, Flatten, Dropout, LSTM, Bidirectional
from keras.models import Sequential
from keras import optimizers
import functions as rnnf
from keras.callbacks import EarlyStopping
from keras.models import load_model
#reloades functions.py module
importlib.reload(rnnf)

#Data preprocessing
processed_data_path = os.path.abspath(os.path.join(os.getcwd(), 'data_for_projects/car_data/processed_data_csv'))

filepath = os.path.join(processed_data_path, 'train_2')
data = pd.read_csv(filepath)

#Choose pid to train LSTM model on
pid_494 = rnnf.pid_for_training(data)

#Create samples and labels for Keras model input
X_train, Y_train = rnnf.matrix_for_rnn_training(pid_494, timestep = 16)

model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True),merge_mode='ave',
                        input_shape=(16, 64)))
model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.1),merge_mode='ave'))
model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.1),merge_mode='ave'))
model.add(Flatten())
#labels for our input data are vectors in 64 dim space hence the last layer
#has to have 64 nodes
model.add(Dense(64, activation = 'hard_sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics =['accuracy'])

#Train the model
history = model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_split = 0.2)

#save trained model
model.save(os.path.join(os.getcwd(), 'Analysis_of_CAN_bus/RNN_anomaly_detection/DNN_models/model_494.HDF5'))
