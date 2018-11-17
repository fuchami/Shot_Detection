# coding: utf-8
"""
抽出した特徴量をLSTMに投げます．投げます

"""

import numpy as np
import argparse

from keras.callbacks import TensorBoard, CSVLogger
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.utils import plot_model
import keras.backend as K
import tensorflow as tf
import h5py

import sys, os, glob
import pandas
import math
import matplotlib.pyplot as plt
from collections import deque

from load import Load_Feature_Data
from model import Models
import tools

def main(args):

    #　保存用ハイパーパラメータ類
    para_str = 'LSTM2_xception_Epoch{}Bachsize{}SeqLength{}Stride{}dropout{}loss{}activation{}_Adam'.format(
            args.epochs,args.batchsize,args.seqlength,args.stride,args.dropout,args.loss)

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

    """ load dataset """
    # ラベル付けしたCSVファイルを読み込む
    load = Load_Feature_Data(args)
    X_train X_validation, Y_train, Y_validation = load.load()

    """ model build """
    md = Model()
    build_model = md.LSTM()
    build_model.summary()

    build_model.compile(loss=args.loss, 
                    optimizer=Adam(), 
                    metrics=([tools.recall,tools.precision,tools.f_score]))

    build_model.fit(X_train, Y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=[tb_cb,csv_logger,reduce_lr],
            validation_data=(X_validation, Y_validation))

    """ model save """
    json_string = build_model.to_json()
    open('./saved_model/'+ para_str + '.json', 'w').write(json_string)
    build_model.save_weights('./saved_weights/'+ para_str +'.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train LSTM for shot detection')
    parser.add_argument('--datasetpath', '-p', type=str, default='./feature_list/xception_feature_list/marge_list.csv')
    parser.add_argument('--epochs', '-e', default=30)
    parser.add_argument('--batchsize', '-b', default=16)
    parser.add_argument('--featurelength', '-f', default=2048)
    parser.add_argument('--seqlength', '-s', default=10)
    parser.add_argument('--loss', '-l', type=str, default='binary_crossentropy')
    parser.add_argument('--dropout', '-d', default=0.3)

    args = parser.parse_args()
    main(args)