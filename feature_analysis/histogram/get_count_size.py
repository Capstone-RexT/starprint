import os
import pandas as pd
import matplotlib.pyplot as plt

def load_packet_sizes(directory):
    packet_sizes = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            data = pd.read_csv(file_path, delimiter="\t", header=None) 
            packet_sizes.extend(data[1].abs()) 
    return packet_sizes

def save_distribution(packet_sizes, save_path):
    packet_sizes = [size for size in packet_sizes if 0 <= size <= 2000]
    
    plt.figure(figsize=(10, 6))

    n, bins, patches = plt.hist(packet_sizes, bins=50, alpha=0.7, color='blue')
    
    plt.title('Packet Size Distribution (0-2000 Bytes)')
    plt.xlabel('Packet Size (bytes)')
    plt.ylabel('frequency + Count')  
    
    for i in range(len(patches)):
        plt.text(patches[i].get_x() + patches[i].get_width() / 2, n[i], int(n[i]), 
                 ha='center', va='bottom', fontsize=8)
    
    plt.xlim(0, 2000)
    
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close() 

directory = '/scratch4/starlink/WFdata75x80/firefox_starlink'
save_directory = '/home/kanghosung/StarlinkWF/data_analysis/size_distribution'

save_file_path = os.path.join(save_directory, 'firefox_starlink_size_distribution_0_to_2000.png')

packet_sizes = load_packet_sizes(directory)
save_distribution(packet_sizes, save_file_path)

print(f'done')
