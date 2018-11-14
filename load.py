# coding: utf-8

"""
データをロードするクラス

データ整形を行うクラス

"""

import numpy as np
import os, sys, csv
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_csv_data(args):
    frame_dir = './'
    X_data = []
    Y_data = []
    seq_length = args.seqlength
    strides = args.strides

    # load csv file
    with open(args.datasetpath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            Y_data.append(int(row[1]))
            
            frame_name = os.path.basename(row[0])
            print("load file:", frame_name)
            img_path = frame_dir + frame_name

            img = load_img(img_path, target_size=(args.imgsize, args.imgsize))
            img_array = img_to_array(img)
            x = (img_array/255.).astype(np.float32)
            print("x.shape", x.shape)

    """ data format """ 



class Load_Feature_Data():

    def __init__(self, seq_length, stride,feature_length):
        
        self.X_data         = []
        self.Y_data         = []
        self.X_             = []
        self.Y_             = []
        self.feature_length = feature_length
        self.seq_length     = seq_length
        self.stride         = stride


    def load(self,feature_path):
        
        with open(feature_path, 'r') as f:

            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                
                self.Y_data.append(int(row[1]))

                feaure = list(map(float, row[2:]))
                self.X_data.append(feaure)

                #self.X_data.astype('float32')
                #self.Y_data.astype('float32')

                # data normalize
                #scaler = MinMaxScaler(feature_range=(0, 1))
                #self.X_data = scaler.fit_transform(self.X_data)
                
        #return (self.X_data, self.Y_data)
        
        """ データ整形， 正規化など """
        ms = MinMaxScaler()
        self.X_data = ms.fit_transform(self.X_data)
        # 時系列の水増し
        length_of_sequence = len(self.X_data) # 全時系列の長さ


        for i in range(0, length_of_sequence - self.seq_length+1, self.stride):
            self.X_.append(self.X_data[i: i + self.seq_length])
            self.Y_.append(self.Y_data[i: i + self.seq_length])

        """ 丁寧に書いた場合
        X_data = np.zeros((len(X), seq_length, features_length), dtype=float)
        Y_data = np.zeros((len(Y), features_length), dtype=float)

        for i, seq in enumerate(X_):
                for t, value in enumerate(seq):
                        for u, feature in enumerate(velue):
                                X_data[i, t, u] = feature
                Y_data[i, 0] = Y_[i]
        """

        self.X_ = np.array(self.X_).reshape(len(self.X_), self.seq_length, self.feature_length)
        self.Y_ = np.array(self.Y_).reshape(len(self.Y_), self.seq_length, 1)
        
        print (self.X_.shape)
        print (self.Y_.shape)

        """ 訓練データと検証データに分割 """
        train_len = int(len(self.X_) * 0.8)
        validation_len = len(self.X_) - train_len

        X_train, X_valid, Y_train, Y_valid =\
                train_test_split(self.X_, self.Y_, test_size=validation_len)
        
        return X_train, X_valid, Y_train, Y_valid
        


