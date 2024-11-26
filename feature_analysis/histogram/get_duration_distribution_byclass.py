import os
import pandas as pd
import matplotlib.pyplot as plt
import math

def get_class_durations(directory):
    class_durations = {}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            classnum = file.split('-')[0]  
            data = pd.read_csv(file_path, delimiter="\t", header=None)
            last_timestamp = data.iloc[-1, 0]  
            

            if classnum not in class_durations:
                class_durations[classnum] = []
            class_durations[classnum].append(last_timestamp)
    
    return class_durations

def save_combined_duration_distribution_by_class(directory, save_base_path):
    class_durations = get_class_durations(directory)
    
    save_path = os.path.join(save_base_path, os.path.basename(directory))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_classes = len(class_durations)
    cols = 9  
    rows = math.ceil(num_classes / cols)  
    
    fig, axs = plt.subplots(rows, cols, figsize=(20, 20)) 
    axs = axs.ravel() 
    
    for i, (classnum, durations) in enumerate(class_durations.items()):
        axs[i].hist(durations, bins=30, alpha=0.75, color='blue', edgecolor='black')
        axs[i].set_title(f"Class {classnum}", fontsize=8)
        axs[i].set_xlabel('Duration', fontsize=6)
        axs[i].set_ylabel('Frequency', fontsize=6)
        axs[i].tick_params(axis='both', which='major', labelsize=6)

    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    graph_save_path = os.path.join(save_path, f"all_classes_cumul_size_distribution.jpg")
    plt.tight_layout()
    plt.savefig(graph_save_path, dpi=300)
    plt.close()
    print(f"Saved all class cumul size graphs in {graph_save_path}")

def process_multiple_directories_for_durations(directories, save_base_path):
    for directory in directories:
        save_combined_duration_distribution_by_class(directory, save_base_path)

directories = [
    '/scratch4/starlink/WFdata75x80/firefox_fiber',
    '/scratch4/starlink/WFdata75x80/firefox_starlink',
    '/scratch4/starlink/WFdata75x80/tor_fiber',
    '/scratch4/starlink/WFdata75x80/tor_starlink'
]


save_base_directory = '/home/kanghosung/StarlinkWF/data_analysis/duration_distribution'

process_multiple_directories_for_durations(directories, save_base_directory)

print('done')
