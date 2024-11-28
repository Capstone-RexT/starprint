import os
import io
import sys
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from keras import backend as K
# from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
import numpy as np
from data_loader import load_dataset_pkl, load_dataset_npz

NB_CLASSES = 75

def load_dataset_split_pkl():
    print("Loading and preparing data for training, and evaluating the model")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset_pkl()
    
    ###########K.set_image_dim_ordering("tf") # tf is tensorflow
    K.set_image_data_format('channels_first')

    # Convert data as float32 type
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_valid = y_valid.astype('float32')
    y_test = y_test.astype('float32')

    # we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
    X_train = X_train[:, :,np.newaxis]
    X_valid = X_valid[:, :,np.newaxis]
    X_test = X_test[:, :,np.newaxis]
    
    print(X_train.shape[0], 'train samples')
    print(X_valid.shape[0], 'validation samples')
    print(X_test.shape[0], 'test samples')

    # Convert class vectors to categorical classes matrices
    y_train = to_categorical(y_train, NB_CLASSES)
    y_valid = to_categorical(y_valid, NB_CLASSES)
    y_test = to_categorical(y_test, NB_CLASSES)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_dataset_split_npz():
    print("Loading and preparing data for training, and evaluating the model")
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset_npz()

    ###########K.set_image_dim_ordering("tf") # tf is tensorflow
    K.set_image_data_format('channels_first')

    # Convert data as float32 type
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    y_train = y_train.astype('float32')
    y_valid = y_valid.astype('float32')
    y_test = y_test.astype('float32')

    # we need a [Length x 1] x n shape as input to the DF CNN (Tensorflow)
    X_train = X_train[:, :,np.newaxis]
    X_valid = X_valid[:, :,np.newaxis]
    X_test = X_test[:, :,np.newaxis]
    
    print(X_train.shape[0], 'train samples')
    print(X_valid.shape[0], 'validation samples')
    print(X_test.shape[0], 'test samples')

    # Convert class vectors to categorical classes matrices
    y_train = to_categorical(y_train, NB_CLASSES)
    y_valid = to_categorical(y_valid, NB_CLASSES)
    y_test = to_categorical(y_test, NB_CLASSES)

    return X_train, X_valid, X_test, y_train, y_valid, y_test