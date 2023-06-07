import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse

def get_filelist(path):
    filelist = []
    file_len = len(path)
    for home, dirs, files in os.walk(path):
        for filename in files:
            file = os.path.join(home, filename)
            if file.endswith(('jpg')):
                label = file.split('/')[-1].split('_')[-1].split('.')[0]
                file = './' + file[file_len:] + ' ' + label
                print("file: ", file)
                filelist.append(file)
    return filelist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True, help='path to dataset')
    args = parser.parse_args()
    filelist = get_filelist(args.image_dir)
    with open('annotation.txt', 'w') as f:
        for file in tqdm(filelist):
            f.write(file+'\n')
    f.close()
