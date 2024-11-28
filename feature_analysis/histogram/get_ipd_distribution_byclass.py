import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_ipd(file_path):
    data = pd.read_csv(file_path, delimiter="\t", header=None)
    timestamps = data[0] 
    ipd_values = timestamps.diff().dropna()  
    return ipd_values

def get_median_ipds_by_class(directory):
    class_median_ipds = [] 
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            ipd_values = calculate_ipd(file_path)
            if len(ipd_values) > 0:
                median_ipd = np.median(ipd_values)
                class_median_ipds.append(median_ipd)
    
    return class_median_ipds


def plot_median_ipd_distribution(directories, output_image_path):
    plt.figure(figsize=(12, 8))  
    
    for idx, directory in enumerate(directories):
        class_median_ipds = get_median_ipds_by_class(directory)
        plt.subplot(2, 2, idx + 1)
        plt.hist(class_median_ipds, bins=30, alpha=0.7, label=os.path.basename(directory))
        plt.title(f'Median IPD Distribution in {os.path.basename(directory)}')
        plt.xlabel('Median IPD')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.ylim(0,20)
    
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.show()

directories = [
    '/scratch4/starlink/WFdata75x80/firefox_fiber',
    '/scratch4/starlink/WFdata75x80/firefox_starlink',
    '/scratch4/starlink/WFdata75x80/tor_fiber',
    '/scratch4/starlink/WFdata75x80/tor_starlink'
]

output_image_path = '/home/kanghosung/StarlinkWF/data_analysis/ipd_stats/median_ipd_distribution_freqlim_20.png'

plot_median_ipd_distribution(directories, output_image_path)

print("done")
