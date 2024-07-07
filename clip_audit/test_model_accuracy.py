
"""
Test accuracy of TinyCLIP on dataset.

Mainly as sanity check and also making sure we're doing our image normalization properly
"""

import os
from clip_audit.utils.load_imagenet import load_imagenet


path = '/network/scratch/s/sonia.joseph/datasets/kaggle_datasets'
dataloader = load_imagenet(path, 'train')



# iterate through
MAX = 5
count = 0

for i, (img, label, idx) in enumerate(dataloader):
    print(img.shape)
    print(label)
    # print(idx)
    count += 1
    if count >= MAX:
        break