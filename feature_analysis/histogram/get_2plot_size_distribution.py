import os
import pandas as pd
import matplotlib.pyplot as plt


def load_packet_sizes(directory):
    positive_packet_sizes = []
    negative_packet_sizes = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            data = pd.read_csv(file_path, delimiter="\t", header=None) 
            

            positive_packet_sizes.extend(data[1][data[1] > 0])
            negative_packet_sizes.extend(data[1][data[1] < 0].abs()) 
    
    return positive_packet_sizes, negative_packet_sizes

def save_distribution(positive_packet_sizes, negative_packet_sizes, save_path):
    # packet filterint (0-2000)
    positive_packet_sizes = [size for size in positive_packet_sizes if 0 <= size <= 2000]
    negative_packet_sizes = [size for size in negative_packet_sizes if 0 <= size <= 2000]
    
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1) 
    n_pos, bins_pos, patches_pos = plt.hist(positive_packet_sizes, bins=50, alpha=0.7, color='green')
    plt.title('Positive Packet Size Distribution (0-2000 Bytes)')
    plt.xlabel('Packet Size (bytes)')
    plt.ylabel('Count')
    
    for i in range(len(patches_pos)):
        plt.text(patches_pos[i].get_x() + patches_pos[i].get_width() / 2, n_pos[i], int(n_pos[i]), 
                 ha='center', va='bottom', fontsize=8)
    
    plt.xlim(0, 2000)
    plt.grid(True)


    plt.subplot(2, 1, 2)  
    n_neg, bins_neg, patches_neg = plt.hist(negative_packet_sizes, bins=50, alpha=0.7, color='red')
    plt.title('Negative Packet Size Distribution (Absolute, 0-2000 Bytes)')
    plt.xlabel('Packet Size (bytes)')
    plt.ylabel('Count')
    
    for i in range(len(patches_neg)):
        plt.text(patches_neg[i].get_x() + patches_neg[i].get_width() / 2, n_neg[i], int(n_neg[i]), 
                 ha='center', va='bottom', fontsize=8)
    
    plt.xlim(0, 2000)
    plt.grid(True)


    plt.tight_layout() 
    plt.savefig(save_path)
    plt.close()  


directory = '/scratch4/starlink/WFdata75x80/firefox_fiber'
save_directory = '/home/kanghosung/StarlinkWF/data_analysis/size_distribution'


save_file_path = os.path.join(save_directory, 'firefox_fiber_size_distribution_pos_neg.png')

positive_packet_sizes, negative_packet_sizes = load_packet_sizes(directory)
save_distribution(positive_packet_sizes, negative_packet_sizes, save_file_path)

print(f'done')
