# coding:utf-8

import sys, os, glob
import matplotlib.pyplot as plt
import argparse

import numpy as np
from keras.callbacks import TensorBoard,EarlyStopping,ReduceLROnPlateau
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import plot_model
import keras.backend as K
import tensorflow as tf
import h5py

import load
import model
import tools

def main(args):

    """ setting """
    para_str = '3dconvNet_Epoch{}_Batchsize{}_SeqLength{}_Stride{}_dropout{}_loss{}_Adam'.format(
        args.epochs, args.batchsize, args.seqlength, args.strides, args.dropout, args.loss)

    """ call back """
    if not os.path.exists('./tb_log/'):
        os.makedirs('./tb_log/')
    tb_cb = TensorBoard(log_dir='./tb_log/'+para_str,
                    histogram_freq=1,
                    write_grads=False,
                    write_graph=True,
                    write_images=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, min_lr=1e-9)
    
    """ load data """
    print("load csv files data")
    #X_train, X_valid, Y_train, Y_valid = load.load_csv_data(args)
    train_datagen = load.ImageDataGenerator()

    """ build model """
    conv3Dmodel = model.conv3D(args)
    conv3Dmodel.summary()
    plot_model(conv3Dmodel, to_file='./images/Conv3DNetworks.png', show_shapes=True)

    conv3Dmodel.compile(loss=args.loss, optimizer=RMSprop())

    """ start train """
    history = conv3Dmodel.fit_generator(
        generator=train_datagen.flow_from_directory(),
        steps_per_epoch=3000,
        epochs=args.epochs,
        callbacks=[tb_cb, reduce_lr],
        verbose=1
    )

    """ model save """
    if not os.path.exists('./saved_model/'):
        os.makedirs('./saved_model/')

    json_string = conv3Dmodel.to_json()
    open('./saved_model/conv3Dmodel.json', 'w').write(json_string)
    conv3Dmodel.save_weights('./saved_model/conv3Dmodel.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train 3Dconv-net for shot detection')
    parser.add_argument('--datasetpath', '-p', type=str, required=True)
    parser.add_argument('--linetoken', '-t', type=str, required=False)
    parser.add_argument('--epochs', '-e', default=300)
    parser.add_argument('--batchsize', '-b', default=32)
    parser.add_argument('--strides', '-s', default=5)
    parser.add_argument('--imgsize', '-i', default=128)
    parser.add_argument('--seqlength', default=10)
    parser.add_argument('--dropout', default=0.3)
    parser.add_argument('--loss', '-l', type=str, default='binary_crossentropy')

    args = parser.parse_args()

    main(args)