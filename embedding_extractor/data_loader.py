# Original Pickel ver.
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import gc

NB_CLASSES = 75

def load_dataset_pkl(feature):
    print("Loading non-defended dataset for closed-world scenario")
    
    dataset_dir = '/scratch4/starlink/baseline/feature/{feature}'

    with open(f'{dataset_dir}/FS_X_{feature},pkl', 'rb') as handle:
        X = pickle.load(handle, encoding='latin1')
        X = np.array(X, dtype=object)
        
    with open(f'{dataset_dir}/FS_y_{feature},pkl', 'rb') as handle:
        y = pickle.load(handle, encoding='latin1')
        y = np.array(y, dtype=object)

    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
    X_valid, X_test, y_valid, y_test=train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    gc.collect()
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_dataset_npz(feature):

    print("Loading non-defended dataset for closed-world scenario")
    dataset_dir = f'/scratch4/kanghosung/starlink_DF/f_ex/{feature}'

    # X represents a sequence of traffic directions
    with open(f'{dataset_dir}/ff_sl_{feature}_training_56inst.npz', 'rb') as handle:
        train = np.load(handle)
        X_train = np.array(train['data'], dtype=object)
        y_train = np.array(train['labels'], dtype=object)
        print(X_train.shape)
        print(X_train[0])

    with open(f'{dataset_dir}/ff_sl_{feature}_valid_12inst.npz', 'rb') as handle:
        valid = np.load(handle)
        X_valid = np.array(valid['data'], dtype=object)
        y_valid = np.array(valid['labels'], dtype=object)
        print(X_valid.shape)
        print(X_valid[0])

    with open(f'{dataset_dir}/ff_sl_{feature}_testing_12inst.npz', 'rb') as handle:
        test = np.load(handle)
        X_test = np.array(test['data'], dtype=object)
        y_test = np.array(test['labels'], dtype=object)
        print(X_test.shape)
        print(X_test[0])

    # X_train, X_test, y_train, y_test=train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # X_valid, X_test, y_valid, y_test=train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape, X_train.ndim)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    y_train = to_categorical(y_train, NB_CLASSES)
    y_valid = to_categorical(y_valid, NB_CLASSES)
    y_test = to_categorical(y_test, NB_CLASSES)

    gc.collect()
    return X_train, y_train, X_valid, y_valid, X_test, y_test
