# coding:utf-8

import tensorflow as tf
import keras
from keras import backend as K

from keras.layers import Dense,Flatten,Dropout,Activation,Input,Conv3D, MaxPooling3D,BatchNormalization
from keras.layers import RepeatVector,Permute,Lambda,merge,multiply,Dot
from keras.layers.recurrent import LSTM,GRU
from keras.models import Sequential, load_model,Model
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.advanced_activations import ELU, LeakyReLU

from surportsPG.custom_recurrents import AttentionDecoder


def conv3D(args):
    # shape = (seqlength, imgsize, imgsize, channels)
    input_shape = (args.seqlength, args.imgsize, args.imgsize, 3)

    model = Sequential()
    # first layer
    model.add(Conv3D(32, (3,3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))

    # second layer
    model.add(Conv3D(64, (3,3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))

    # 3rd layer
    model.add(Conv3D(128, (3,3,3), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))
    # 4th layer
    model.add(Conv3D(256, (2,2,2), activation='relu'))
    model.add(Conv3D(256, (2,2,2), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(args.dropout))
    model.add(Dense(512))
    model.add(Dropout(args.dropout))
    model.add(Dense(1, activation='softmax'))

    return model


class Models():

    def __init__(self,seq_length,feature_length):

        self.n_hidden       = 1024
        self.n_hidden2      = 512
        self.feature_length = feature_length
        self.seq_length     = seq_length
        self.dropout        = 0.3
        self.units          = 64

    def LSTM(self):

        """ LSTM 1層 """
        with tf.name_scope('LSTM_model') as scope:
            model = Sequential()
            
            with tf.name_scope('LSTM') as scope:
                model.add(LSTM(self.n_hidden,
                        input_dim=self.feature_length,
                        input_length= self.seq_length,
                        return_sequences=True))
            
            with tf.name_scope('Dense') as scope:
                model.add(TimeDistributed(Dense(1, activation='linear')))
        
        return model

    def LSTM2(self):

        """ LSTM 2層 """
        with tf.name_scope('LSTM model') as scope:
            model = Sequential()
            
            with tf.name_scope('LSTM1') as scope:
                model.add(LSTM(self.n_hidden,
                        input_dim=self.feature_length,
                        input_length= self.seq_length,
                        return_sequences=True))
            
            with tf.name_scope('LSTM2') as scope:
                model.add(LSTM(self.n_hidden,
                        input_dim=self.feature_length,
                        input_length= self.seq_length,
                        return_sequences=True))

            with tf.name_scope('Dense') as scope:
                model.add(TimeDistributed(Dense(1, activation='linear')))
        
        return model

    def attention_3d_block(self, inputs):
        print(inputs.shape)
        input_dim = int(inputs.shape[2]) 
        a = Permute((2, 1))(inputs)
        a = TimeDistributed(Dense(self.seq_length, activation='softmax'))(a)
        a_probs = Permute((2,1), name = 'attention_vec')(a)
        output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
        return output_attention_mul

    def Attention_before_LSTM(self):
       _input = Input(shape=(self.seq_length, self.feature_length)) 

       drop1 = Dropout(0.3)(_input)
       attention_mul = self.attention_3d_block(drop1)

       attention_mul = LSTM(self.n_hidden, return_sequences=True)(attention_mul)
       output = TimeDistributed(Dense(1, activation='linear'))(attention_mul)

       model = Model(input=[_input], output=output)

       return model


    def Attention_after_LSTM(self):
        _input = Input(shape=(self.seq_length, self.feature_length))

        drop = Dropout(0.3)(_input)

        LSTM_layer = LSTM(self.n_hidden, return_sequences=True)(drop)
        attention_mul = self.attention_3d_block(LSTM_layer)

        drop2 = Dropout(0.3)(attention_mul)
        output = TimeDistributed(Dense(1, activation='linear'))(drop2)

        model = Model(input=[_input], output=output)

        return model

    def Attention_LSTM(self):

        _input = Input(shape=(self.seq_length, self.feature_length))

        dropout = Dropout(0.3)(_input)

        LSTM_layer = Bidirectional(LSTM(self.n_hidden, return_sequences=True))(dropout)

        # Attention layer
        #attention = TimeDistributed(Dense(1, activation='tanh'))(LSTM_layer)
        #attention = Flatten()(attention)
        #attention = Activation('softmax')(attention)
        #attention = RepeatVector(self.n_hidden)(attention)
        #attention = Permute([2,1])(attention)

        #sent_representation = multiply([LSTM_layer, attention], name='attention_mul')
        #sent_representation = Lambda(lambda xin: K.sum(xin, axis=2), output_shape=(self.units,))(sent_representation)

        attention_mul = self.attention_3d_block(LSTM_layer)
        #attention_flatten = Flatten()(attention_mul)

        dropout2 = Dropout(0.3)(attention_mul)
        output = TimeDistributed(Dense(1, activation='linear'))(dropout2)

        #probabilities = TimeDistributed(Dense(1, activation=self.activation))(sent_representation)

        model = Model(input=_input, outputs=output)
        return model

    def simpleNMT(self):

        input_ = Input(shape=(self.seq_length, self.feature_length))

        lstm = Bidirectional(LSTM(self.n_hidden, return_sequences=True),
                                name='bidirectional_1',
                                merge_mode = 'concat',
                                trainable=True)(input_)
        
        y_hat =  TimeDistributed(AttentionDecoder(self.n_hidden,
                                name='attention',
                                output_dim = 1,
                                return_probabilities=False,
                                trainable=True))(lstm)
        
        model = Model(inputs=input_, outputs=y_hat)
        return model




