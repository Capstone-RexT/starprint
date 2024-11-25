#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Last Modified: 2024-11-13
# Modified By: H. Kang

import tensorflow as tf
from keras import backend as K
import random
import pickle
from keras.utils import np_utils
from keras.optimizers import adamax_v2
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser
import numpy as np
import os
import gc
import logging
from datetime import datetime
import sys
from utility import LoadDataNoDefCW
from Model_NoDef import DFNet

random.seed(0)

# Argument Parser 설정
PARSER = ArgumentParser()
PARSER.add_argument("-feature1", type=str, default="ipd", help="First feature name")
PARSER.add_argument("-feature2", type=str, default="size", help="Second feature name")
PARSER.add_argument("-feature3", type=str, default="direction", help="Third feature name")
PARSER.add_argument("--apply_scaler", action="store_true", help="Apply Standard Scaler to the data.")
PARSER.add_argument("--use_early_stopping", action="store_true", help="Use Early Stopping during training.")
PARSER.add_argument("--use_smoothing", action="store_true", help="Enable temperature scaling for smoothing.")
PARSER.add_argument("-g", type=str, default='7')
PARSER.add_argument("-e", type=int, default='30')
args = PARSER.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = args.g

# 인자값 설정
FEATURE1 = args.feature1
FEATURE2 = args.feature2
FEATURE3 = args.feature3
SCENARIO = "ff_sl"
NB_EPOCH = args.e
BATCH_SIZE = 128
LENGTH = 5000
NB_CLASSES = 100
INPUT_SHAPE = (LENGTH, 1)
OPTIMIZER = adamax_v2.Adamax(lr=0.002)

# 로그 설정
log_path = f"/scratch4/kanghosung/starlink_DF/log/weightedvoting/{FEATURE1}_{FEATURE2}_{FEATURE3}"
if not os.path.exists(log_path):
    os.makedirs(log_path, exist_ok=True)

# 로그 파일명 생성 함수
def get_log_filename(feature1, feature2, feature3, epoch, apply_scaler, early_stopping, accuracy=None):
    suffix = "_SS" if apply_scaler else ""
    suffix += "_ES" if early_stopping else ""
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if accuracy is not None:
        return f"{feature1}_{feature2}_{feature3}_{epoch}{suffix}_{current_time}({accuracy:.2f}).txt"
    return f"{feature1}_{feature2}_{feature3}_{epoch}{suffix}_{current_time}.txt"

# LoggerWriter 클래스 정의
class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.level(message)

    def flush(self):
        pass

# Logger 설정 함수
def create_log(log_path, feature1, feature2, feature3, epoch, apply_scaler, early_stopping):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_filename = get_log_filename(feature1, feature2, feature3, epoch, apply_scaler, early_stopping)
    log_full_path = os.path.join(log_path, log_filename)
    
    file_handler = logging.FileHandler(log_full_path, 'w')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    
    print(f"Log is being saved at: {log_full_path}")
    log.info(f"Log files are being saved at: {log_full_path}")
    return log

log = create_log(log_path, FEATURE1, FEATURE2, FEATURE3, NB_EPOCH, args.apply_scaler, args.use_early_stopping)
sys.stdout = LoggerWriter(log, log.info)

# 데이터 로드 및 전처리
def load_data(feature, scenario, apply_scaler=False):
    root = f"/scratch4/kanghosung/starlink_DF/f_ex/{feature}"
    tr_path = os.path.join(root, f"{scenario}_{feature}_training_56inst.npz")
    val_path = os.path.join(root, f"{scenario}_{feature}_valid_12inst.npz")
    te_path = os.path.join(root, f"{scenario}_{feature}_testing_12inst.npz")
    
    train = np.load(tr_path)
    valid = np.load(val_path)
    test = np.load(te_path)

    X_train, y_train = train['data'], train['labels']
    X_valid, y_valid = valid['data'], valid['labels']
    X_test, y_test = test['data'], test['labels']

    if apply_scaler:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# 데이터 전처리 함수
def preprocess_data(X, y):
    X = X.astype('float32')[:, :, np.newaxis]
    y = np_utils.to_categorical(y, NB_CLASSES)
    return X, y

X_train1, y_train, X_valid1, y_valid, X_test1, y_test = load_data(FEATURE1, SCENARIO, args.apply_scaler)
X_train2, _, X_valid2, _, X_test2, _ = load_data(FEATURE2, SCENARIO, args.apply_scaler)
X_train3, _, X_valid3, _, X_test3, _ = load_data(FEATURE3, SCENARIO, args.apply_scaler)

X_train1, y_train = preprocess_data(X_train1, y_train)
X_valid1, y_valid = preprocess_data(X_valid1, y_valid)
X_test1, y_test = preprocess_data(X_test1, y_test)

# 모델 빌드 및 학습
model1 = DFNet.build(INPUT_SHAPE, NB_CLASSES).compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
model2 = DFNet.build(INPUT_SHAPE, NB_CLASSES).compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
model3 = DFNet.build(INPUT_SHAPE, NB_CLASSES).compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)] if args.use_early_stopping else []

model1.fit(X_train1, y_train, epochs=NB_EPOCH, validation_data=(X_valid1, y_valid), callbacks=callbacks)
model2.fit(X_train2, y_train, epochs=NB_EPOCH, validation_data=(X_valid2, y_valid), callbacks=callbacks)
model3.fit(X_train3, y_train, epochs=NB_EPOCH, validation_data=(X_valid3, y_valid), callbacks=callbacks)

# 모델 학습
log.info("Training model 1...")
model1.fit(X_train1, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=2,
           validation_data=(X_valid1, y_valid), callbacks=callbacks)

log.info("Training model 2...")
model2.fit(X_train2, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=2,
           validation_data=(X_valid2, y_valid), callbacks=callbacks)

log.info("Training model 3...")
model3.fit(X_train2, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=2,
           validation_data=(X_valid2, y_valid), callbacks=callbacks)
           
# Validation Accuracy 기반 가중치 계산
val_acc1 = model1.evaluate(X_valid1, y_valid, verbose=0)[1]
val_acc2 = model2.evaluate(X_valid2, y_valid, verbose=0)[1]
val_acc3 = model3.evaluate(X_valid3, y_valid, verbose=0)[1]

def get_model_weights(val_acc1, val_acc2, val_acc3):
    total = val_acc1 + val_acc2 + val_acc3
    return val_acc1 / total, val_acc2 / total, val_acc3 / total

def weighted_voting(X_test1, X_test2, X_test3):
    prob1, prob2, prob3 = model1.predict(X_test1), model2.predict(X_test2), model3.predict(X_test3)
    w1, w2, w3 = get_model_weights(val_acc1, val_acc2, val_acc3)
    averaged_prob = w1 * prob1 + w2 * prob2 + w3 * prob3
    return np.argmax(averaged_prob, axis=1)

y_pred = weighted_voting(X_test1)
accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
#log.info(f"Weighted Voting Testing Accuracy: {accuracy:.4f}")

# 로그 파일명 업데이트
#update_log_filename(log_path, FEATURE1, FEATURE2, FEATURE3, NB_EPOCH, args.apply_scaler, args.use_early_stopping, accuracy * 100)
