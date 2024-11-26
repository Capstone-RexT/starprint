import os
import pandas as pd
import random

def find_bursts(x):
    direction = x[0]
    bursts = []
    start = 0
    temp_burst = x[0]
    for i in range(1, len(x)):
        if x[i] == 0.0:
            break
        elif x[i] == direction:
            temp_burst += x[i]
        else:
            if abs(temp_burst) >= 5:  # filtering : burst length 5~
                bursts.append((start, i, temp_burst))
            start = i
            temp_burst = x[i]
            direction *= -1
    return bursts


def get_burst_statistics(directory):
    burst_statistics = {} 
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            data = pd.read_csv(file_path, delimiter="\t", header=None)
            direction_column = data[1].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0) 
            
            bursts = find_bursts(direction_column.tolist())
            if bursts:
                burst_statistics[file_path] = bursts
            
    return burst_statistics

def save_burst_statistics(directories, save_path):
    with open(save_path, 'w') as f:
        for directory in directories:
            burst_stats = get_burst_statistics(directory)

            selected_files = random.sample(list(burst_stats.keys()), min(5, len(burst_stats)))

            f.write(f"Directory: {directory}\n")
            for file_path in selected_files:
                bursts = burst_stats[file_path][:20] 
                f.write(f"File: {file_path}\n")
                for start, end, burst in bursts:
                    f.write(f"  Start: {start}, End: {end}, Burst: {burst}\n")
            f.write("\n")

    print(f"Burst statistics saved to {save_path}")

directories = [
    '/scratch4/starlink/WFdata75x80/firefox_fiber',
    '/scratch4/starlink/WFdata75x80/firefox_starlink',
    '/scratch4/starlink/WFdata75x80/tor_fiber',
    '/scratch4/starlink/WFdata75x80/tor_starlink'
]

save_directory = '/home/kanghosung/StarlinkWF/data_analysis/packet_stats'
save_file_path = os.path.join(save_directory, 'all_burst_stats.txt')

save_burst_statistics(directories, save_file_path)

print(f'done')
