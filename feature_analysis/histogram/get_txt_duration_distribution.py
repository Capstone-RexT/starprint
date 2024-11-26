import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

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

def save_statistics_by_class(directory, save_base_path):
    class_durations = get_class_durations(directory)
    
    save_path = os.path.join(save_base_path, os.path.basename(directory))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    stats_file = os.path.join(save_path, 'class_duration_statistics.txt')
    
    variances = []
    stddevs = []
    kurts = []
    skewnesses = []
    
    with open(stats_file, 'w') as f:
        for classnum, durations in class_durations.items():
            variance = np.var(durations)
            stddev = np.std(durations)
            kurt = kurtosis(durations)
            skewness = skew(durations)
            
            variances.append(variance)
            stddevs.append(stddev)
            kurts.append(kurt)
            skewnesses.append(skewness)
            
            f.write(f"Class {classnum} Statistics:\n")
            f.write(f"  Variance: {variance:.4f}\n")
            f.write(f"  Standard Deviation: {stddev:.4f}\n")
            f.write(f"  Kurtosis: {kurt:.4f}\n")
            f.write(f"  Skewness: {skewness:.4f}\n")
            f.write("\n")
        
        f.write("Overall Statistics:\n")
        
        # 분산
        f.write(f"Variance - Min: {np.min(variances):.4f}, Max: {np.max(variances):.4f}, Mean: {np.mean(variances):.4f}, Median: {np.median(variances):.4f}\n")
        # 표준편차
        f.write(f"Standard Deviation - Min: {np.min(stddevs):.4f}, Max: {np.max(stddevs):.4f}, Mean: {np.mean(stddevs):.4f}, Median: {np.median(stddevs):.4f}\n")
        # 첨도
        f.write(f"Kurtosis - Min: {np.min(kurts):.4f}, Max: {np.max(kurts):.4f}, Mean: {np.mean(kurts):.4f}, Median: {np.median(kurts):.4f}\n")
        # 왜도
        f.write(f"Skewness - Min: {np.min(skewnesses):.4f}, Max: {np.max(skewnesses):.4f}, Mean: {np.mean(skewnesses):.4f}, Median: {np.median(skewnesses):.4f}\n")

    print(f"Saved statistics for directory {directory} in {stats_file}")

def process_multiple_directories_for_statistics(directories, save_base_path):
    for directory in directories:
        save_statistics_by_class(directory, save_base_path)

directories = [
    '/scratch4/starlink/WFdata75x80/firefox_fiber',
    '/scratch4/starlink/WFdata75x80/firefox_starlink',
    '/scratch4/starlink/WFdata75x80/tor_fiber',
    '/scratch4/starlink/WFdata75x80/tor_starlink'
]

save_base_directory = '/home/kanghosung/StarlinkWF/data_analysis/txt_duration_statistics'


process_multiple_directories_for_statistics(directories, save_base_directory)

print('done')
