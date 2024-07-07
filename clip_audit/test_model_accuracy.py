
"""
Test accuracy of TinyCLIP on dataset.

Mainly as sanity check and also making sure we're doing our image normalization properly
"""

import os
from clip_audit.utils.load_imagenet import load_imagenet

import argparse


def main(imagenet_path, model_type, model_name):

    path = '/network/scratch/s/sonia.joseph/datasets/kaggle_datasets'
    dataloader = load_imagenet(path, 'train')



    # Check original model
    if model_type == 'original':
        model = torch.hub.load(model_name, 'ViT-B/32', jit=False)
    elif model_type == 'hooked':
        model = torch.hub.load('openai/clip', 'ViT-B/32', jit=False)
        model = hook_model(model)

    # iterate through
    MAX = 5
    count = 200

    with torch.no_grad():
        for i, (img, label, idx) in enumerate(dataloader):
            print(img.shape)
            print(label)
            # print(idx)
            count += 1
            if count >= MAX:
                break

    


# Results are

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Test accuracy of TinyCLIP on dataset')
    parser.add_argument('--imagenet_path', type=str, default='/network/scratch/s/sonia.joseph/datasets/kaggle_datasets', help='Path to dataset')
    parser.add_argument('--model_type', type=str, default='original', help='Model to test (original/hooked)")
    parser.add_argument('--model_name', type=str, default='openai/clip', help='Model name to test')

    args = parser.parse_args()
    
    main(args.imagenet_path, args.model_type, args.model_name)