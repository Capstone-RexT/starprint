# %%
import os
import sys
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
from tensorflow.keras.optimizers.legacy import Adamax
from huggingface_hub import login, hf_hub_download
from safetensors.torch import load_file
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import gc
from tqdm import tqdm
from models import Llama3Model
from functions import rescale_theta, model_memory_size
from splitter import load_dataset_split_npz, load_dataset_split_pkl

torch.manual_seed(123)
login("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")

# Set to use only one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

feature = "ipd_npz"
quantile = True
scale_factor = 10000
# %%
# Hyperparameters
NB_EPOCH = 30
BATCH_SIZE = 128
VERBOSE = 1
LENGTH = 5000
OPTIMIZER = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
NB_CLASSES = 75
INPUT_SHAPE = (LENGTH, 1)

# LLAMA Configuration
LLAMA32_CONFIG = {
    "vocab_size": 128_256,      # Vocabulary size
    "context_length": 131_072,  # Context length
    "emb_dim": 2048,            # Embedding dimension
    "n_heads": 32,              # Number of attention heads
    "n_layers": 16,             # Number of layers
    "hidden_dim": 8192,         # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,           # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,     # The base in RoPE's "theta"
    "dtype": torch.bfloat16,    # Lower-precision dtype to save memory
    "rope_freq": {              # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}
LLAMA_SIZE_STR = "1B"

# Adjust context length and RoPE theta
old_context_length = LLAMA32_CONFIG["context_length"]
LLAMA32_CONFIG["context_length"] = 8192
LLAMA32_CONFIG["rope_base"] = rescale_theta(LLAMA32_CONFIG["rope_base"], old_context_length, LLAMA32_CONFIG["context_length"])

print(f"Updated RoPE theta: {LLAMA32_CONFIG['rope_base']}")

# Initialize model
llama = Llama3Model(LLAMA32_CONFIG)

# Model parameter count
total_params = sum(p.numel() for p in llama.parameters())
print(f"Total parameters: {total_params:,}")

# Account for weight tying
total_params_normalized = total_params - llama.tok_emb.weight.numel()
print(f"Unique parameters (after weight tying): {total_params_normalized:,}")

# Model memory size
print(f"Memory usage (float32): {model_memory_size(llama, input_dtype=torch.float32):.2f} GB")
print(f"Memory usage (bfloat16): {model_memory_size(llama, input_dtype=torch.bfloat16):.2f} GB")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
llama.to(device)

# Load datasets
X_train, y_train, X_valid, y_valid, X_test, y_test = load_dataset_split_npz(feature[0:-4])
if quantile:
    qt = QuantileTransformer(output_distribution='normal', random_state=0)
    X_train = qt.fit_transform(X_train)
    X_valid = qt.fit_transform(X_valid)
    X_test = qt.fit_transform(X_test)

vocab_size = llama.module.tok_emb.num_embeddings if torch.cuda.device_count() > 1 else llama.tok_emb.num_embeddings
# Continue with embedding and training steps as in original code
# Embedding extraction with progress bar
for ((data, target), desc) in zip(
    [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)],
    ["train", "valid", "test"]
):
    embedding_list = []
    label_list = []
    for (input_ids, y_idx) in tqdm(zip(data, target), total=data.shape[0], desc=f"Processing {desc}"):
        input_ids = input_ids.reshape(-1)

        if scale_factor > 1:
            input_ids = input_ids * scale_factor
            min_value = min(input_ids)
            input_ids -= min_value

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)  # Ensure tensor is on the same device
        input_ids = torch.clamp(input_ids, max=vocab_size - 1)

        # print(input_ids)
        # print(f"Input IDs shape: {input_ids.shape}, Max value: {input_ids.max()}, Min value: {input_ids.min()}")
        input_ids -= input_ids.min()
        # Move model outputs to CPU if necessary
        embedding_vector = llama(input_ids, output_embeddings=True).to("cpu")

        # Append embeddings
        embedding_list.append(embedding_vector.detach().to(torch.float32).numpy())
        label_list.append(y_idx)

        # Garbage collection and memory cleanup after each batch
        del input_ids
        del embedding_vector
        gc.collect()

    # Convert lists to numpy arrays and save after processing each dataset
    embedding_array = np.array(embedding_list)
    label_array = np.array(label_list)

    # Save embeddings to npy files
    np.save(f"embedding/{feature}/X_{desc.lower()}_{LLAMA_SIZE_STR}_{feature}{scale_factor}_embeddings_quantile.npy", embedding_array)
    np.save(f"embedding/{feature}/y_{desc.lower()}_{LLAMA_SIZE_STR}_{feature}{scale_factor}_embeddings_quantile.npy", label_array)

    # Clear lists to free memory for next iteration
    embedding_list.clear()
    label_list.clear()
