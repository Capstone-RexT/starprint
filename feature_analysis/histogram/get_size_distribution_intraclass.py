import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_packet_sizes(file_path):
    data = pd.read_csv(file_path, delimiter="\t", header=None)
    packet_sizes = data[1].abs()  
    return packet_sizes

def get_median_packet_sizes_by_class(directory):
    class_median_sizes = []  
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            packet_sizes = calculate_packet_sizes(file_path)
            if len(packet_sizes) > 0:
                median_size = np.median(packet_sizes)
                class_median_sizes.append(median_size)
    
    return class_median_sizes

def plot_median_packet_size_distribution(directories, output_image_path):
    plt.figure(figsize=(12, 8))  
    
    for idx, directory in enumerate(directories):
        class_median_sizes = get_median_packet_sizes_by_class(directory)
        plt.subplot(2, 2, idx + 1)  
        plt.hist(class_median_sizes, bins=30, alpha=0.7, label=os.path.basename(directory))
        plt.title(f'Median Packet Size Distribution in {os.path.basename(directory)}')
        plt.xlabel('Median Packet Size (bytes)')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        #plt.ylim(0, 20) 
    
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.show()

directories = [
    '/scratch4/starlink/WFdata75x80/firefox_fiber',
    '/scratch4/starlink/WFdata75x80/firefox_starlink',
    '/scratch4/starlink/WFdata75x80/tor_fiber',
    '/scratch4/starlink/WFdata75x80/tor_starlink'
]

output_image_path = '/home/kanghosung/StarlinkWF/data_analysis/size_distribution/median_packet_size_distribution_intraclass.png'

plot_median_packet_size_distribution(directories, output_image_path)

print("done")
