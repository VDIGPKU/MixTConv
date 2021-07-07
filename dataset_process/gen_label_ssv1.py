# Code for MixTConv: Mixed Temporal Convolutional Kernels for Efficient Action Recognition
# arXiv: https://arxiv.org/abs/2001.06769
# Kaiyu Shan
# shankyle@pku.edu.cn
# ------------------------------------------------------
# Code adapted from https://github.com/metalbubble/TRN-pytorch/blob/master/process_dataset.py
# processing the raw data of the video Something-Something-V1
# file path: -decompression_path
#            ----something
#            -----v1
# Aug.21 2019

import os

if __name__ == '__main__':
    decompression_path = '/data/shankaiyu/something/v1/'  # decompression of the raw data and *.csv
    dataset_name = 'something-something-v1'  # 'jester-v1'
    with open(decompression_path + '%s-labels.csv' % dataset_name) as f:
        lines = f.readlines()
    categories = []
    for line in lines:
        line = line.rstrip()
        categories.append(line)
    categories = sorted(categories)
    with open(decompression_path + 'category.txt', 'w') as f:
        f.write('\n'.join(categories))

    dict_categories = {}
    for i, category in enumerate(categories):
        dict_categories[category] = i

    files_input = [decompression_path + '%s-validation.csv' % dataset_name,
                   decompression_path + '%s-train.csv' % dataset_name]
    files_output = [decompression_path + 'val_videofolder.txt', decompression_path + 'train_videofolder.txt']
    for (filename_input, filename_output) in zip(files_input, files_output):
        with open(filename_input) as f:
            lines = f.readlines()
        folders = []
        idx_categories = []
        for line in lines:
            line = line.rstrip()
            items = line.split(';')
            folders.append(items[0])
            idx_categories.append(dict_categories[items[1]])
        output = []
        for i in range(len(folders)):
            curFolder = folders[i]
            curIDX = idx_categories[i]
            # counting the number of frames in each video folders
            dir_files = os.listdir(os.path.join(decompression_path + 'img', curFolder))
            output.append('%s %d %d' % ('something/v1/img/' + curFolder, len(dir_files), curIDX))
            print('%d/%d' % (i, len(folders)))
        with open(filename_output, 'w') as f:
            f.write('\n'.join(output))
