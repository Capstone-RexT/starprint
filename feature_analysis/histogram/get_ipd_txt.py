import os
import pandas as pd
import numpy as np


def calculate_ipd(file_path):
    data = pd.read_csv(file_path, delimiter="\t", header=None)
    timestamps = data[0] 
    ipd_values = timestamps.diff().dropna()  

    return ipd_values


def get_ipd_statistics(directory):
    folder_ipd_stats = {} 
    
    for root, dirs, files in os.walk(directory):
        all_ipd = []
        for file in files:
            file_path = os.path.join(root, file)
            ipd_values = calculate_ipd(file_path)
            all_ipd.extend(ipd_values)
        
        if all_ipd: 
            folder_ipd_stats[root] = all_ipd
    
    return folder_ipd_stats

def save_ipd_statistics(directory_list, save_path):
    with open(save_path, 'w') as f:
        for directory in directory_list:
            folder_ipd_stats = get_ipd_statistics(directory)
            for folder, ipd_values in folder_ipd_stats.items():
                min_ipd = np.min(ipd_values)
                max_ipd = np.max(ipd_values)
                mean_ipd = np.mean(ipd_values)
                median_ipd = np.median(ipd_values)
                
                f.write(f"{folder}\n")
                f.write(f"Total Files: {len(ipd_values)}\n")
                f.write(f"Min IPD: {min_ipd}\n")
                f.write(f"Max IPD: {max_ipd}\n")
                f.write(f"Mean IPD: {mean_ipd:.6f}\n")
                f.write(f"Median IPD: {median_ipd:.6f}\n\n")
    
    print(f"IPD statistics saved to {save_path}")


def process_multiple_directories(directories, save_path):
    save_ipd_statistics(directories, save_path)

directories = [
    '/scratch4/starlink/WFdata75x80/firefox_fiber',
    '/scratch4/starlink/WFdata75x80/firefox_starlink',
    '/scratch4/starlink/WFdata75x80/tor_fiber',
    '/scratch4/starlink/WFdata75x80/tor_starlink'
]

save_directory = '/home/kanghosung/StarlinkWF/data_analysis/ipd_stats'
save_file_path = os.path.join(save_directory, 'all_ipd_stats.txt')


process_multiple_directories(directories, save_file_path)

print("done")
