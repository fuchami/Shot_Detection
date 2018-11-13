# coding: utf-8

"""
抽出した特徴量をLSTMに投げる．いわば実行だ

"""

import os
import numpy as np

import keras
from keras.optimizers import Adam, SGD,RMSprop
from keras.callbacks import CSVLogger,TensorBoard,EarlyStopping,ReduceLROnPlateau
from keras.utils.vis_utils import plot_model

from load import Load_Feature_Data
from model import Models
import tools as t

class LSTMs(object):

    def __init__(self):

        """ 各パラメータの初期化"""
        self.args          = ''
        self.FEATURE_PATH  = './feature_list/fc2_feature_list/marge_list.csv' 
        self.tb_log        = './tb_log/'
        self.epochs        = 300
        self.batchsize     = 32
        self.featue_length = 4096
        self.seq_length    = 10 # ひとつの時系列の長さ
        self.stride        = 1 
        self.loss         = "mean_squared_error"
        #self.loss          = "binary_crossentropy"
        self.activation    = 'linear'
        self.kanji_op      = SGD(lr=1e-2, decay=1e-9, momentum=0.9, nesterov=True)
        self.opt           = Adam()
        self.para_str      = 'Attention_after_LSTM_xception_Epoch{}Bachsize{}SeqLength{}Stride{}activation{}_loss{}'.format(
                                                self.epochs,self.batchsize,self.seq_length,self.stride,self.activation,self.loss)

    def build(self):
        md = Models(self.seq_length,self.featue_length)
        model = md.Attention_after_LSTM()

        plot_model(model, to_file='Attention_after_LSTM_model.png', show_shapes=True)

        return model

    def train(self):
        """ build model """
        model = self.build()
        model.summary()

        """ dataset load """
        #print(os.path.exists(self.FEATURE_PATH))
        load = Load_Feature_Data(self.seq_length,self.stride,self.featue_length)
        X_train, X_validation, Y_train, Y_validation = load.load(self.FEATURE_PATH) 

        """ callback """
        csv_logger = CSVLogger('./csv_log/'+self.para_str+ '.csv', separator=',')
        tb_cb = TensorBoard(log_dir='./tb_log/'+ self.para_str,
            histogram_freq=1,
            write_grads=True,
            write_graph=False,
            write_images=1)
        es_cb = EarlyStopping (monitor='val_loss', patience=0, verbose=1, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,verbose=1, min_lr=1e-6)

        model.compile(loss=self.loss,
                    optimizer=self.opt,
                    metrics=[t.recall, t.precision, t.f_score])

        model.fit(X_train, Y_train,
                batch_size=self.batchsize,
                epochs=self.epochs,
                callbacks=[tb_cb,csv_logger,reduce_lr],
                validation_data=(X_validation,Y_validation))

        """ model save """
        json_string = model.to_json()
        open('./saved_model/'+ self.para_str + '.json', 'w').write(json_string)
        model.save_weights('./saved_weights/'+ self.para_str +'.h5')

if __name__ == '__main__' :

    lstm = LSTMs()

    lstm.train()







