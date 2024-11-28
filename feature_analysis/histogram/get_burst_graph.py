import os
import pandas as pd
import matplotlib.pyplot as plt

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
            if abs(temp_burst) >= 5:
                bursts.append((start, i, temp_burst))
            start = i
            temp_burst = x[i]
            direction *= -1
    return bursts

def get_burst_statistics(directory):
    burst_statistics = [] 
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            data = pd.read_csv(file_path, delimiter="\t", header=None)
            direction_column = data[1].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0) 
            
            bursts = find_bursts(direction_column.tolist())
            if bursts:
                burst_statistics.extend(bursts)
    
    return burst_statistics

def generate_combined_burst_heatmap(directories, save_path):
    plt.figure(figsize=(15, 10))

    max_timestamp = 0 
    burst_data_per_directory = {}

    for directory in directories:
        burst_data = get_buarst_statistics(directory)
        burst_df = pd.DataFrame(burst_data, columns=['Start', 'End', 'Burst Length'])
 
        if not burst_df.empty:
            max_timestamp = max(max_timestamp, burst_df['End'].max())
        
        burst_counts = burst_df.groupby('Start').size().reset_index(name='Count')
        burst_data_per_directory[directory] = burst_counts

    for i, (directory, burst_counts) in enumerate(burst_data_per_directory.items()):
        plt.subplot(2, 2, i + 1)

        plt.fill_between(burst_counts['Start'], burst_counts['Count'], alpha=0.5)
        plt.title(f'Burst Frequency for {os.path.basename(directory)}')
        plt.xlabel('Time')
        plt.ylabel('Count of Bursts')
        plt.xlim(0, min(max_timestamp, 1500)) 
        plt.ylim(0, min(burst_counts['Count'].max() + 1, 400))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'combined_burst_graph.jpg'))
    plt.close()

directories = [
    '/scratch4/starlink/WFdata75x80/firefox_fiber',
    '/scratch4/starlink/WFdata75x80/firefox_starlink',
    '/scratch4/starlink/WFdata75x80/tor_fiber',
    '/scratch4/starlink/WFdata75x80/tor_starlink'
]

save_directory = '/home/kanghosung/StarlinkWF/data_analysis/packet_stats'

generate_combined_burst_heatmap(directories, save_directory)

print(f'done')
