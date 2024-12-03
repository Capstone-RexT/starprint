import numpy as np
from natsort import natsorted
import glob
import argparse
from functions import *

parser = argparse.ArgumentParser()
parser.add_argument("-f", '--feature', type=str, default="TAM1D")
parser.add_argument("-o", '--output', type=str, default="/scratch4/starlink/baseline")
parser.add_argument("-i", '--input', type=str, default="/scratch4/starlink/WFdata75x80")
parser.add_argument("-b", '--browser', type=str, default="firefox")
parser.add_argument("-l", '--link', type=str, default="starlink")

INPUT_SIZE = 5000
SITE = 75

feature_dict = {
    'Tiktok': get_tiktok,
    'direction': get_direction,
    'TAM1D': get_tam1d,
    'IPD': get_ipd
}

def get_feature(feature, instance):
    feature_func = feature_dict.get(feature)
    return feature_func(instance)

def get_metadata(file_name):
    site, instance = map(int, file_name.split('/')[-1].split('.')[0].split('_')[0].split('-'))
    return site, instance

def extract(input_path, output_path, feature, browser, link):
    tp = "FF"
    if browser == "firefox": 
        if link == "starlink": tp = "FS"
        else: tp = "FF"
    elif browser == "tor": 
        if link == "starlink": tp = "TS"
        else: tp = "TF"
    
    print(f"Extract {feature} from {tp} ({browser} on {link})")

    X = []
    y = []
    for file in natsorted(glob.glob(f"{input_path}/{browser}_{link}/*-*")):
        site, instance = map(int, file.split("/")[-1].split("-"))
        with open(file) as f:
            X.append(get_feature(feature, f.readlines()))
        y.append(site)
    
    X = np.array(X)
    y = np.array(y)

    print(X.shape, y.shape)

    np.savez(f'{output_path}/{feature}_{tp}_data.npz', data=X, index=y)

    print("Extract process has successfully done")

if __name__ == "__main__":
    args = parser.parse_args()
    extract(args.input, args.output, args.feature, args.browser, args.link)