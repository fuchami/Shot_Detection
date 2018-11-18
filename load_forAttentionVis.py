# coding: utf-8

"""
データをロードするクラス
データ整形を行うクラス

attention_visualize用のロードクラス

"""

import numpy as np
import os, sys, csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class Load_Feature_Data():

    def __init__(self, seq_length, stride,feature_length):
        
        self.X_data         = []
        self.Y_data         = []
        self.X_             = []
        self.Y_             = []
        self.feature_length = feature_length
        self.seq_length     = seq_length
        self.stride         = stride
        self.img_path_list  = []
        self.path_list      = []

        self.XX             = []
        self.YY             = []
        self.path           = []


    def load(self,feature_path):
        
        with open(feature_path, 'r') as f:

            reader = csv.reader(f)
            header = next(reader)

            for row in reader:
                
                self.img_path_list.append(row[0])
                
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
            self.path_list.append(self.img_path_list[i: i + self.seq_length])


        self.X_ = np.array(self.X_).reshape(len(self.X_), self.seq_length, self.feature_length)
        self.Y_ = np.array(self.Y_).reshape(len(self.Y_), self.seq_length, 1)
        self.path_list= np.array(self.path_list).reshape(len(self.path_list), self.seq_length, 1)
        
        print (self.X_.shape)
        print (self.Y_.shape)
        print (self.path_list.shape)

        X_ = self.X_ 
        Y_ = self.Y_
        path_list = self.path_list

        # random
        #permutation = np.random.permutation(len(X_))
        #self.X_  = [X_[n] for n in permutation]
        #self.Y_ = [Y_[n] for n in permutation]
        #self.path_list = [path_list[n] for n in permutation]

        # 10番目が1のデータをたくさん集める
        for i in range(0, len(X_)):
            if self.Y_[i][10] == 1:
                #XX.append.(self.X_[i])
                self.XX.append(self.X_[i])
                self.YY.append(self.Y_[i])
                self.path.append(self.path_list[i])

        print (len(self.XX) )
        return self.XX, self.YY, self.path
        
        


