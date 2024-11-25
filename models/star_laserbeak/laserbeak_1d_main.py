#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Last Modified: 2024-11-18
# Modified By: H. Kang

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import os
import gc
import logging
from datetime import datetime
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from argparse import ArgumentParser
from src.transdfnet import DFNet  # Laserbeak 모델 사용

# Argument parser for command line options
PARSER = ArgumentParser()
PARSER.add_argument("--apply_scaler", action="store_true", help="Apply Standard Scaler to the data.")
PARSER.add_argument("--quantile_trans", action="store_true", help="Apply Quantile Transformer to the data.")
PARSER.add_argument("-g", type=str, default='6')
PARSER.add_argument("-f", type=str, default="ipd")
args = PARSER.parse_args()

# Set random seed and CUDA device
random.seed(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = args.g
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE = args.f
SCENARIO = "ff_sl"
print("Feature:", FEATURE)

# Training parameters
NB_EPOCH = 40
BATCH_SIZE = 128
LENGTH = 5000
NB_CLASSES = 100
INPUT_SHAPE = (LENGTH, 1)
LEARNING_RATE = 0.002

# Set up log path and configuration
log_path = "/scratch4/kanghosung/starlink_DF/log/laserbeak"
if not os.path.exists(log_path):
    os.makedirs(log_path)
    print(f"Log path '{log_path}' created.")

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.level(message)

    def flush(self):
        pass

def create_log(log_path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_{current_time}.txt"
    file_handler = logging.FileHandler(os.path.join(log_path, log_filename), 'w')
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    return log

log = create_log(log_path)
sys.stdout = LoggerWriter(log, log.info)

def load_data(feature, apply_scaler=False, quantile_trans=False):
    root = f"/scratch4/kanghosung/starlink_DF/f_ex/{feature}"
    tr_path = os.path.join(root, f"{SCENARIO}_{feature}_training_56inst.npz")
    val_path = os.path.join(root, f"{SCENARIO}_{feature}_valid_12inst.npz")
    te_path = os.path.join(root, f"{SCENARIO}_{feature}_testing_12inst.npz")

    train = np.load(tr_path)
    valid = np.load(val_path)
    test = np.load(te_path)

    X_train, y_train = train['data'], train['labels']
    X_valid, y_valid = valid['data'], valid['labels']
    X_test, y_test = test['data'], test['labels']

    # Apply scaling or transformation
    if apply_scaler:
        print("Applying Standard Scaler...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
    elif quantile_trans:
        print("Applying Quantile Transformer...")
        transformer = QuantileTransformer(output_distribution='normal', random_state=0)
        X_train = transformer.fit_transform(X_train)
        X_valid = transformer.transform(X_valid)
        X_test = transformer.transform(X_test)

    X_train = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32)
    X_valid = torch.tensor(X_valid[:, np.newaxis, :], dtype=torch.float32)
    X_test = torch.tensor(X_test[:, np.newaxis, :], dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_valid = torch.tensor(y_valid, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

print("Loading and preparing data")
X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(FEATURE, args.apply_scaler, args.quantile_trans)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)

print("Building and training Laserbeak DF model")
model = DFNet(num_classes=NB_CLASSES, input_channels=1).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(model, loader):
    model.train()
    total_loss, correct = 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == y).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

def evaluate(model, loader):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == y).sum().item()

    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

for epoch in range(NB_EPOCH):
    train_loss, train_acc = train(model, train_loader)
    val_loss, val_acc = evaluate(model, valid_loader)
    log.info(f"Epoch {epoch+1}/{NB_EPOCH}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

test_loss, test_acc = evaluate(model, test_loader)
log.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
