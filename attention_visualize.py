# coding:utf-8

"""
Attentionの可視化

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.layers import merge
from keras.layers.core import *
from keras.models import *
import keras.backend as K

from load_test import Load_Feature_Data


def get_activations(model, inputs, print_shape_only=False, layer_name=None):

    print('--------activations--------')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)

    return activations

if __name__ == '__main__':
    
    feature_path = './feature_list/fc2_feature_list/marge_list.csv'  
    seq_length = 20
    feature_length = 4096
    stride = 3
    # load trained model
    para_str = "Attention_after_BidiLSTM_fc2_Epoch100Bachsize64SeqLength20Stride3activationlinear_lossmean_squared_errordrop0.5"

    print ("model build !!")
    model = model_from_json(open('./saved_model/' + para_str + '.json', 'r').read())
    print ("model weight load !!")
    model.load_weights('./saved_weights/' + para_str + '.h5')

    model.summary()
    
    attention_vectors = []

    # load test data
    print ("test data load !")
    load = Load_Feature_Data(seq_length, stride, feature_length)
    X_val, Y_val, path_list = load.load(feature_path=feature_path)

    for i in range(300):
        X_sample = X_val[i]
        #print (Y_val[i])
        #print(X_sample.shape)
        print(path_list[i])
        X_sample = np.array(X_sample).reshape(1,seq_length, feature_length)

        activations =get_activations(model, X_sample, print_shape_only=True, layer_name='attention_vec')
        attention_vector = np.var(activations[0], axis=2).squeeze()

        print ('attention = ', attention_vector)
        assert (np.sum(attention_vector) - 1.0) < 1e-5
        attention_vectors.append(attention_vector)
    
    attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    #attention_vector_final = np.array(attention_vector)

    pd.DataFrame(attention_vector_final, columns=['attention (%)']).plot(kind='bar',
                                                                            title='Attention Mechanism as '
                                                                            'a function of input'
                                                                            ' dimensions.')
    plt.show() 
    