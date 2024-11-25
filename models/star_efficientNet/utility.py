#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Last Modified: 2024-11-20
# Modified By: H. Kang

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


# ================================================
class NLUDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, normalization=False):
        self.data = data
        self.labels = labels
        self.normalization = normalization

    def normalize(self, x):
        mu = np.mean(x)
        sigma = np.std(x)
        return (x - mu) / sigma

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.normalization:
            x = self.normalize(x)
        x = np.expand_dims(x, axis=0)  # Add channel dimension
        label = self.labels[idx]
        return x, label


# ================================================
def evaluate(dataloader, device, net, loss_fn):
    net.eval()
    n, loss, score = 0., 0., 0.
    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device, dtype=torch.float)
            target = target.to(device)
            logits = net(x)
            loss += (loss_fn(logits, target).item()) * x.size(0)
            score += (logits.argmax(dim=1) == target).float().sum().item()
            n += x.size(0)
        loss /= n
        score /= n
    return score, loss

def calculate_dynamic_scaling_params(input_size, num_classes):
    alpha = 1 + (input_size / 10000) 
    beta = 1 + (num_classes / 100)
    phi = (alpha + beta) / 2

    return alpha, beta, phi


# ================================================
def save_ckpt(ckpt_path, net, best_validation_acc):
    ckpt = {'net': net.state_dict(),
            'best_validation_acc': best_validation_acc}
    torch.save(ckpt, ckpt_path)
    return 'Checkpoint saved!'


def load_ckpt(ckpt_path, net):
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        try:
            net.load_state_dict(ckpt['net'])
            best_validation_acc = ckpt['best_validation_acc']
            print(f'Checkpoint loaded (Best validation accuracy: {best_validation_acc:.4f})')
        except RuntimeError:
            print('Invalid checkpoint format.')
    else:
        raise ValueError('No checkpoint exists.')
    return net, best_validation_acc


def save_train_log(result_dir, train_losses, valid_losses, valid_accs, best_validation_acc):
    with open(os.path.join(result_dir, 'trainloss.txt'), 'w') as f:
        f.write(str(train_losses))
    with open(os.path.join(result_dir, 'validloss.txt'), 'w') as f:
        f.write(str(valid_losses))
    with open(os.path.join(result_dir, 'validacc.txt'), 'w') as f:
        f.write(str(valid_accs))

    plt.figure(dpi=300, figsize=(6, 6))

    plt.subplot(211)
    plt.plot(list(train_losses.keys()), list(train_losses.values()), label='Train Loss')
    plt.plot(list(valid_losses.keys()), list(valid_losses.values()), label='Validation Loss')
    plt.legend()
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')

    plt.subplot(212)
    plt.plot(list(valid_accs.keys()), list(valid_accs.values()), label='Validation Accuracy')
    plt.grid()
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.2])
    plt.title(f'Validation Accuracy (Best: {best_validation_acc:.4f})')

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'training_results.png'))
    plt.close('all')
