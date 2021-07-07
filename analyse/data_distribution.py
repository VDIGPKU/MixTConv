# Code for MixTConv: Mixed Temporal Convolutional Kernels for Efficient Action Recognition
# arXiv: https://arxiv.org/abs/2001.06769
# Kaiyu Shan
# shankyle@pku.edu.cn
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V1
# file path: -decompression_path
#            ----something
#            ------v1
# Aug.21 2019

import os
import pandas as pd

if __name__ == '__main__':
    decompression_path = '/data/shankaiyu/something/v1/'  # decompression of the raw data and *.csv
    dataset_name = 'something-something-v1'  # 'jester-v1'
    with open(decompression_path + '%s-labels.csv' % dataset_name) as f:
        lines = f.readlines()
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)

    files_input = [decompression_path + '%s-validation.csv' % dataset_name,
                   decompression_path + '%s-train.csv' % dataset_name]
    files_output = ['val_distribution.csv', 'train_distribution.csv']
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            lines = f.readlines()
        dict_count = {}
        for category in categories:
            dict_count[category] = 0
        for line in lines:
            line = line.rstrip()
            items = line.split(';')
            dict_count[items[1]] = dict_count[items[1]] + 1

        pd.DataFrame.from_dict(dict_count, orient='index', columns=['video number']).to_csv(filename_output)

