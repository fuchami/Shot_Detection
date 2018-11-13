# coding: utf-8

"""
抽出した特徴量をLSTMに投げます．投げます


"""



import numpy as np

from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.utils import plot_model
import keras.backend as K
import tensorflow as tf

import sys, os, glob
import matplotlib.pyplot as plt
import pandas
import math
import matplotlib.pyplot as plt
from collections import deque

from load import Load_Feature_Data
from model import Models

# ラベル付けしたCSVファイルを読み込む
load = Load_Feature_Data(self.seq_length, self.stride, self.feature_length)
FEATURE_PATH = './feature_list/xception_feature_list/marge_list.csv'

epochs          = 1
batch_size      = 16
features_length = 2048
seq_length      = 10 #ひとつの時系列の長さ
n_hidden1       = 1024
n_hidden2       = 512
stride          = 5
dropout         = 0.
loss            = 'binary_crossentropy'
activation      = 'sigmoid'
input_shape     = (seq_length, features_length)
opt = Adam()


#　保存用ハイパーパラメータ類
para_str = 'LSTM2_xception_Epoch{}Bachsize{}SeqLength{}Stride{}dropout{}loss{}activation{}_Adam'.format(epochs,batch_size,seq_length,stride,dropout,loss,activation)


"""  Call back 定義 """
csv_logger = CSVLogger('./csv_log/'+para_str+ '.csv', separator=',')

tb_cb = TensorBoard(log_dir='./TensorBoard_log/'+ para_str,
            histogram_freq=1,
            write_grads=True,
            write_graph=True,
            write_images=1
            )
es_cb = EarlyStopping (monitor='val_loss', patience=0, verbose=1, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,verbose=1, min_lr=1e-9)

""" f score define """
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# 検出率　TP / (TP + FN)
def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# fscore
def f_score(y_true, y_pred, beta=1):
        if beta < 0:
                raise ValueError('The lowest choosable beta is zero (only precision).')
        
        if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
                return 0
        
        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        bb = beta ** 2
        f_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
        return f_score

# データ読み込み
X_train X_validation, Y_train, Y_validation = load.load(FEATURE_PATH)
"""
# 正規化
ms = MinMaxScaler()
X = ms.fit_transform(X)

# データ整形，時系列情報を増やします
length_of_sequences = len(X) #全時系列の長さ
print (len(Y))

X_ = []
Y_ = []

for i in range(0, length_of_sequences - seq_length+1, stride):
        X_.append(X[i: i + seq_length])

        Y_.append(Y[i: i + seq_length])


X_data = np.array(X_).reshape(len(X_), seq_length, features_length)
Y_data = np.array(Y_).reshape(len(Y_), seq_length, 1)

print (X_data.shape)
print (Y_data.shape)

# train and test
train_len = int(len(X_data) * 0.8)
validarion_len = len(X_data) - train_len

X_train, X_validation, Y_train, Y_validation = \
        train_test_split(X_data, Y_data, test_size=validarion_len)
"""

#model build
md = Model()
build_model = md.LSTM()
build_model.summary()

build_model.compile(loss=loss, 
                optimizer=opt, 
                metrics=([recall,precision,f_score]))


build_model.fit(X_train, Y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tb_cb,csv_logger,reduce_lr],
        validation_data=(X_validation, Y_validation))

# model save
json_string = build_model.to_json()
open('./saved_model/'+ para_str + '.json', 'w').write(json_string)
build_model.save_weights('./saved_weights/'+ para_str +'.h5')


