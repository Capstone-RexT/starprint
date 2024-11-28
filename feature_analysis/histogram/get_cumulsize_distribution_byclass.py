import os
import pandas as pd
import matplotlib.pyplot as plt
import math


def get_class_cumul_sizes(directory):
    class_cumul_sizes = {}  
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            classnum = file.split('-')[0]
            data = pd.read_csv(file_path, delimiter="\t", header=None) 
            cumul_size = data[1].abs().sum()
            
            if 'tor_fiber' in root or 'tor_firefox' in root:
                cumul_size *= 512
            
            if classnum not in class_cumul_sizes:
                class_cumul_sizes[classnum] = []
            class_cumul_sizes[classnum].append(cumul_size)
    
    return class_cumul_sizes

def save_cumul_size_distribution_by_class(directory, save_base_path):
    class_cumul_sizes = get_class_cumul_sizes(directory)
    
    save_path = os.path.join(save_base_path, os.path.basename(directory))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    num_classes = len(class_cumul_sizes)
    rows = math.ceil(num_classes / 10)  
    cols = min(10, num_classes) 
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    fig.tight_layout(pad=3.0)
    
    for idx, (classnum, cumul_sizes) in enumerate(class_cumul_sizes.items()):
        row_idx = idx // 10
        col_idx = idx % 10
        
        ax = axes[row_idx, col_idx] if rows > 1 else axes[col_idx]
        ax.hist(cumul_sizes, bins=30, alpha=0.75, color='blue', edgecolor='black')
        ax.set_title(f"Class {classnum}", fontsize=10)
        ax.set_xlabel('Cumulative Size', fontsize=8)
        ax.set_ylabel('Frequency', fontsize=8)
    
    if len(class_cumul_sizes) < rows * cols:
        for j in range(len(class_cumul_sizes), rows * cols):
            fig.delaxes(axes.flatten()[j])

    graph_save_path = os.path.join(save_path, f"all_classes_cumul_size_distribution.jpg")
    plt.savefig(graph_save_path, dpi=300)
    plt.close()
    print(f"Saved all class cumul size graphs in {graph_save_path}")

def process_multiple_directories_for_cumul_sizes(directories, save_base_path):
    for directory in directories:
        save_cumul_size_distribution_by_class(directory, save_base_path)


directories = [
    '/scratch4/starlink/WFdata75x80/firefox_fiber',
    '/scratch4/starlink/WFdata75x80/firefox_starlink',
    '/scratch4/starlink/WFdata75x80/tor_fiber',
    '/scratch4/starlink/WFdata75x80/tor_starlink'
]

save_base_directory = '/home/kanghosung/StarlinkWF/data_analysis/size_distribution/cumul_size_distribution_byclass'

process_multiple_directories_for_cumul_sizes(directories, save_base_directory)

print('done')
