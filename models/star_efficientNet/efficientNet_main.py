#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Last Modified: 2024-11-20
# Modified By: H. Kang

import os
import sys
import argparse
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utility import *
from cnn_1d_adaptive import CNN1d_adaptive, AdaptiveDFNet
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from torchsummary import summary

# ================================================
# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--apply_scaler", action="store_true", help="Apply Standard Scaler")
parser.add_argument("--quantile_trans", action="store_true", help="Apply Quantile Transformer")
parser.add_argument("--use_early_stopping", action="store_true", help="Use Early Stopping during training")
parser.add_argument("-g", type=str, default="7", help="GPU device to use")
parser.add_argument("-e", type=int, default=100, help="Number of training epochs")
parser.add_argument("-f", type=str, default="ipd", help="Feature name")
args = parser.parse_args()

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = args.g
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global parameters
FEATURE = args.f
SCENARIO = "ff_sl"
NB_EPOCH = args.e
BATCH_SIZE = 128
NUM_CLASSES = 100
LOG_PATH = f"/scratch4/kanghosung/starlink_EfficientNet/log/{FEATURE}"


# ================================================
def setup_logger(log_path, feature, epoch, apply_scaler, quantile_trans, early_stopping):
    def get_log_filename():
        suffix = "_SS" if apply_scaler else ""
        suffix += "_QT" if quantile_trans else ""
        if early_stopping:
            suffix += "_ES"
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{feature}_{epoch}{suffix}_{current_time}.txt"

    os.makedirs(log_path, exist_ok=True)
    log_filename = os.path.join(log_path, get_log_filename())

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_filename, 'w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    sys.stdout = LoggerWriter(logger, logger.info)
    sys.stderr = LoggerWriter(logger, logger.error)

    logger.info(f"Logging to {log_filename}")
    return logger, log_filename


class LoggerWriter:
    """
    Custom logger to redirect stdout and stderr to a logger.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.level(message)

    def flush(self):
        pass


# ================================================
def load_data(feature, scenario, apply_scaler=False, quantile_trans=False):
    root = f"/scratch4/kanghosung/starlink_DF/f_ex/{feature}"
    tr_path = os.path.join(root, f"{scenario}_{feature}_training_56inst.npz")
    val_path = os.path.join(root, f"{scenario}_{feature}_valid_12inst.npz")
    te_path = os.path.join(root, f"{scenario}_{feature}_testing_12inst.npz")
    # Load datasets
    train = np.load(tr_path)
    valid = np.load(val_path)
    test = np.load(te_path)

    X_train, y_train = train["data"], train["labels"]
    X_valid, y_valid = valid["data"], valid["labels"]
    X_test, y_test = test["data"], test["labels"]

    if apply_scaler:
        print("Applying Standard Scaler...")
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    elif quantile_trans:
        print("Applying Quantile Transformer...")
        transformer = QuantileTransformer(output_distribution="normal", random_state=42)
        X_train = transformer.fit_transform(X_train)
        X_valid = transformer.transform(X_valid)
        X_test = transformer.transform(X_test)

    print(f"Data Loaded: Training={X_train.shape}, Validation={X_valid.shape}, Testing={X_test.shape}")
    return X_train, y_train, X_valid, y_valid, X_test, y_test


# ================================================
def main():
    logger, log_filename = setup_logger(LOG_PATH, FEATURE, NB_EPOCH, args.apply_scaler, args.quantile_trans, args.use_early_stopping)

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(FEATURE, SCENARIO, args.apply_scaler, args.quantile_trans)

    alpha, beta, phi = calculate_dynamic_scaling_params(X_train.shape[1], NUM_CLASSES)

    logger.info(f"Dynamic Scaling Parameters - Alpha: {alpha:.2f}, Beta: {beta:.2f}, Phi: {phi:.2f}")

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                                   torch.tensor(y_train, dtype=torch.long))
    valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float32).unsqueeze(1),
                                   torch.tensor(y_valid, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1),
                                  torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model setup
    #model = CNN1d_adaptive(kernel_size=3, num_classes=NUM_CLASSES, alpha=alpha, beta=beta, phi=phi).to(device)
    model = AdaptiveDFNet(input_shape=(1, X_train.shape[1]), num_classes=NUM_CLASSES, 
                         alpha=alpha, beta=beta, phi=phi, kernel_size=8).to(device)
    #kernel 4, 6, 12, 16 으로 실험적으로 결정?
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    summary(model, input_size=(1, 5000))

    best_valid_acc = 0
    train_losses, valid_losses, valid_accs = {}, {}, {}

    # Training loop
    for epoch in range(NB_EPOCH):
        logger.info(f"Epoch {epoch + 1}/{NB_EPOCH}")

        # Train one epoch
        model.train()
        train_loss = 0
        correct = 0  # To track correct predictions
        total = 0    # To track total samples

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1) 
            correct += (predicted == targets).sum().item() 
            total += targets.size(0) 

        train_loss /= len(train_loader.dataset)
        train_acc = correct / total
        train_losses[epoch] = train_loss

        # Validation
        valid_acc, valid_loss = evaluate(valid_loader, device, model, criterion)
        valid_losses[epoch] = valid_loss
        valid_accs[epoch] = valid_acc

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            checkpoint_path = f"/scratch4/kanghosung/starlink_EfficientNet/model/{FEATURE}/best_model.pt"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            logger.info(f"Saving checkpoint to {checkpoint_path} with accuracy {best_valid_acc:.4f}")
            save_ckpt(checkpoint_path, model, best_valid_acc)

        scheduler.step()

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")

    save_train_log(LOG_PATH, train_losses, valid_losses, valid_accs, best_valid_acc)

    # Test
    checkpoint_path = f"/scratch4/kanghosung/starlink_EfficientNet/model/{FEATURE}/best_model.pt"
    model, _ = load_ckpt(f"/scratch4/kanghosung/starlink_EfficientNet/model/{FEATURE}/best_model.pt", model)
    test_acc, _ = evaluate(test_loader, device, model, criterion)
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Log saved at: {log_filename}")


if __name__ == "__main__":
    main()
