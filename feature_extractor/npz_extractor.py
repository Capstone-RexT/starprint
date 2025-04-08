import os
import re
import numpy as np
import pandas as pd
import glob
import argparse
from tqdm import tqdm
from natsort import natsorted
from extractors import *

# extractor 함수 리스트
extractor_functions = [
    extract_burst,
    extract_direction,
    extract_ipd, 
    extract_ipd_filtered,
    extract_cumulative_size,
    extract_size,
    extract_1dtam,
    extract_upload_tam,
    extract_download_tam,
]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some folders and rule names.')
    parser.add_argument('--input-path', type=str, default="WFdata75x80/firefox_starlink", help='Root folder for input features')
    parser.add_argument('--output-path', type=str, default="filtered", help='Path to the output NPZ file')
    parser.add_argument('--csv-path', type=str, default="result.csv", help='Path to the csv file that contains the average accuracy')
    parser.add_argument('--start-site', type=int, default=0, help='Starting index of the site_code to use')
    parser.add_argument('--classNum', type=int, default=75, help='Number of classes to use')
    parser.add_argument('--start-instance', type=int, default=68, help='Starting index of instances to use per class')
    parser.add_argument('--instanceNum', type=int, default=12, help='Number of instances to use per class')
    return parser.parse_args()

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_top_classes(csv_path):
    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values(by='Avg acc', ascending=False)
    return {
        'top75': df_sorted.head(75)['class'].astype(str).tolist(),
        'top70': df_sorted.head(70)['class'].astype(str).tolist(),
        'top65': df_sorted.head(65)['class'].astype(str).tolist(),
        'top60': df_sorted.head(60)['class'].astype(str).tolist(),
        'top50': df_sorted.head(50)['class'].astype(str).tolist()
    }

def process_instance_file(file_path):
    with open(file_path, 'r') as f:
        instance = [line.split('\t') for line in f.readlines()]
    return [(float(t), int(s)) for t, s in instance]

def format_feature_name(func):
    if isinstance(func, type(lambda: 0)):
        raw_name = func.__code__.co_name
    else:
        raw_name = func.__name__
    
    # 함수 이름에서 'extract' 접두사 제거하고, CamelCase를 snake_case로 변환
    name = re.sub(r'^extract_?', '', raw_name)  # extract_ 또는 extract 제거
    return name

def extract_features_to_ndarray(traces, label, func):
    try:
        result = func([traces])[0]
    except Exception as e:
        print(f"  [ERROR] {func.__name__} failed: {e}")
        return None

    if len(result) < 5000:
        result += [0] * (5000 - len(result))    # list
    else:
        result = result[:5000]

    result.append(label)
    return np.array(result, dtype=np.float32)

def save_features_to_npz(features, save_path):
    features = np.array(features)
    if len(features) == 0:
        print(f"  [WARNING] No data to save at {save_path}")
        return
    features = features[np.argsort(features[:, -1])]
    X = features[:, :-1]
    y = features[:, -1].astype(np.int64)
    np.savez(save_path, data=X.astype(np.float32), labels=y)
    print(f"  Saved: {save_path}")

def process_feature_group(group_name, class_list, input_path, output_path, class_labels):
    print(f"\n[+] Processing group: {group_name}")
    group_dir = os.path.join(output_path, group_name)

    for func in extractor_functions:
        func_name = func.__name__ if not isinstance(func, type(lambda: 0)) else func.__code__.co_name
        feature_name = format_feature_name(func)

        print(f"  [*] Extracting with {func_name}")
        all_data, train_data, valid_data, test_data = [], [], [], []

        for class_str in tqdm(class_list, desc=f"    → {feature_name}", unit="class"):
            class_label = class_labels.get(class_str)
            files = natsorted(glob.glob(os.path.join(input_path, f"{class_str}-*")))

            if len(files) < 80:
                print(f"    [-] Skipping {class_str} (only {len(files)} instances)")
                continue

            for idx, file in enumerate(files[:80]):
                traces = process_instance_file(file)
                feature = extract_features_to_ndarray(traces, class_label, func)
                if feature is None:
                    continue
                all_data.append(feature)
                if idx < 56:
                    train_data.append(feature)
                elif idx < 68:
                    valid_data.append(feature)
                else:
                    test_data.append(feature)

        func_out_dir = os.path.join(group_dir, feature_name)
        ensure_directory_exists(func_out_dir)
        save_features_to_npz(all_data, os.path.join(func_out_dir, f"ff_sl_{feature_name}_all_80inst.npz"))
        save_features_to_npz(train_data, os.path.join(func_out_dir, f"ff_sl_{feature_name}_training_56inst.npz"))
        save_features_to_npz(valid_data, os.path.join(func_out_dir, f"ff_sl_{feature_name}_valid_12inst.npz"))
        save_features_to_npz(test_data, os.path.join(func_out_dir, f"ff_sl_{feature_name}_testing_12inst.npz"))

if __name__ == "__main__":
    args = parse_arguments()

    # 클래스 레이블 매핑
    class_labels = {}
    with open("firefox_fiber_site_instance_counts.txt", 'r') as f:
        for idx, line in enumerate(f.readlines()):
            site_code = line.split(',')[0].strip()
            class_labels[site_code] = idx

    class_groups = load_top_classes(args.csv_path)

    for group_name, class_list in class_groups.items():
        process_feature_group(group_name, class_list, args.input_path, args.output_path, class_labels)