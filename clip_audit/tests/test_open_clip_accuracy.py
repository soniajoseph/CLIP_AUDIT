'''
File to test the accuracy of OpenCLIP.
'''

import open_clip

list_of_models = open_clip.list_pretrained()

for model in list_of_models:
    print(model)
    

# Command to evaluate a model
# python -m open_clip_train.main \
#     --imagenet-val /network/datasets/imagenet.var/imagenet_torchvision/val \
#     --model ViT-B-32-256 \
#     --pretrained datacomp_s34b_b86k

