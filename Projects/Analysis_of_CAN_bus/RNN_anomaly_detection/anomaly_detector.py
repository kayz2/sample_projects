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

processed_data_path = os.path.abspath(os.path.join(os.getcwd(), 'data_for_projects/car_data/processed_data_csv'))
filepath = os.path.join(processed_data_path, 'test_data')
data_test = pd.read_csv(filepath_test)
pid_494_test = rnnf.pid_for_training(data_test)
X_test, Y_test = rnnf.matrix_for_rnn_training(pid_494_test, timestep = 16)

#Load already trained model
model = load_model(os.path.join(os.getcwd(), 'Analysis_of_CAN_bus/RNN_anomaly_detection/DNN_models/model_494.HDF5'))

#Use trained model to make prediction on the train dataset
pred_test = model.predict(X_test)

#compute mu and standard deviation of the prediction error
#Those will be used for our anomaly detector
diff = pred_test - Y_test
nor = np.linalg.norm(diff, axis =1)
mu = nor.mean()
sigma = nor.std()

#load dataset with anomalies
filepath_anomaly = os.path.join(processed_data_path, 'data_for_anomalies')
data_anomalies = pd.read_csv(filepath_anomaly)
pid_494_anomaly = rnnf.pid_for_training(data_anomalies)
X_anomaly, Y_anomaly = rnnf.matrix_for_rnn_training(pid_494_with_anomaly, timestep = 16)

pred_anomaly = model.predict(X_anomaly)

diff_anomaly = pred_anomaly - Y_anomaly
error_anomaly = np.linalg.norm(diff_anomaly, axis =1)

#Create p-value based anomaly detector of the prediction error
anomaly_df = pd.DataFrame(error_anomaly, columns = ['error'])
anomaly_df = anomaly_df.iloc[16:]
anomaly_df = anomaly_df.assign(rolling_error_5 = anomaly_df.error.rolling(5).mean())
#use already computed mu and sigma to compute z-score
anomaly_df = anomaly_df.assign(z_score = anomaly_df.rolling_error_5.apply(lambda z: (z - mu)/sigma ))
anomaly_df = anomaly_df.assign(p_val = anomaly_df.z_score.apply(lambda z: st.norm.cdf(z)))

#create visual of the p-values
p_value = anomaly_df['p_val'].tolist()
p_value = p_value[5:]
fig = plt.plot(p_value)

#keep p-value greater than .995 or smaller than .005 (those are signals that
#are unlikely to occur) the rest gets mapped to .5
anomalous_signals = []
for i in range(len(p_value)):
     if p_value[i] < .005:
         test.append(p_value[i])
     elif p_value[i] > .995:
         test.append(p_value[i])
     else:
         test.append(.5)
         pass

fig_2 = plt.plot(anomalous_signals)
