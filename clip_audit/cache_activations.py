
"""
Test accuracy of TinyCLIP on dataset.

Mainly as sanity check and also making sure we're doing our image normalization properly
"""

import os
from clip_audit.utils.load_imagenet import load_imagenet, get_imagenet_names

import argparse
from transformers import CLIPProcessor, CLIPModel
import torch

from vit_prisma.models.base_vit import HookedViT

from torchvision import transforms


from tqdm.auto import tqdm
import numpy as np
import h5py



def create_save_path(save_dir, model_name, train_val, layer_type):
    return os.path.join(save_dir, model_name, train_val, f"{layer_type}.h5")

def main(imagenet_path, train_val, model_name, save_dir, total_layers=12):
    ind_to_name, imagenet_names = get_imagenet_names(imagenet_path)
    
    # Note -- change this to get CLIP processor
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])

    dataloader = load_imagenet(imagenet_path, train_val, shuffle=False, transform=transform)

    device = 'cuda'

    model = HookedViT.from_pretrained(model_name, is_timm=False, is_clip=True, fold_ln=False, center_writing_weights=False) # in future, do all models
    model.to(device)
    print(f"Model loaded: {model_name}")


    # Make sure these are indeed the right layers.
    # Create names_filter
    hook_point_names = []
    for layer in range(12):  # Assuming 12 layers
        hook_point_names.extend([
            f"blocks.{layer}.hook_mlp_out",
            f"blocks.{layer}.hook_resid_post",
            f"blocks.{layer}.mlp.hook_post"
        ])
    names_filter = lambda name: name in hook_point_names


    layer_types = ["hook_mlp_out", "hook_resid_post", "hook_post"]
    # Create a file for each layer type
    files = {layer_type: h5py.File(create_save_path(save_dir, model_name, train_val, layer_type), 'w') 
             for layer_type in layer_types}


    image_index = 0
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            output, cache = model.run_with_cache(images, names_filter=names_filter)

            # if cache is empty, show warning
            if not cache:
                print(f"Warning: cache is empty for image index {image_index}.")

            for layer, activation in cache.items():
                activation = activation.cpu().numpy()

                # Determine which file to save to based on the layer name
                for layer_type in layer_types:
                    if layer_type in layer:
                        f = files[layer_type]
                        break
                else:
                    print(f"Warning: Unknown layer type for layer {layer}. Skipping.")
                    continue

                # Save activation
                if layer not in f:
                    f.create_dataset(layer, data=activation, maxshape=(None, *activation.shape[1:]), chunks=True)
                    if 'image_indices' not in f:
                        f.create_dataset('image_indices', data=np.arange(image_index, image_index + images.shape[0]), maxshape=(None,), chunks=True)
                else:
                    current_size = f[layer].shape[0]
                    f[layer].resize((current_size + activation.shape[0]), axis=0)
                    f[layer][-activation.shape[0]:] = activation
                    
                    f['image_indices'].resize((current_size + images.shape[0],))
                    f['image_indices'][-images.shape[0]:] = np.arange(image_index, image_index + images.shape[0])

            image_index += images.shape[0]

    # Close all files
    for f in files.values():
        f.close()

    print(f"Finished caching activations and image indices for {image_index} images.")


    # with h5py.File(save_path, 'w') as f:
    #     image_index = 0
    #     with torch.no_grad():
    #         for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
    #             images = images.to(device)
    #             labels = labels.to(device)

    #             output, cache = model.run_with_cache(images, names_filter=names_filter)

    #             # if cache is empty, show warning
    #             if not cache:
    #                 print(f"Warning: cache is empty for image index {image_index}.")

    #             for layer, activation in cache.items():
    #                 activation = activation.cpu().numpy()

    #                 # activation shape
    #                 if layer not in f:
    #                     f.create_dataset(layer, data=activation, maxshape=(None, *activation.shape[1:]), chunks=True)
    #                     if 'image_indices' not in f:
    #                         f.create_dataset('image_indices', data=np.arange(image_index, image_index + images.shape[0]), maxshape=(None,), chunks=True)
    #                 else:
    #                     current_size = f[layer].shape[0]
    #                     f[layer].resize((current_size + activation.shape[0]), axis=0)
    #                     f[layer][-activation.shape[0]:] = activation
                        
    #                     f['image_indices'].resize((current_size + images.shape[0],))
    #                     f['image_indices'][-images.shape[0]:] = np.arange(image_index, image_index + images.shape[0])

    #             image_index += images.shape[0]

    #     print(f"Finished caching activations and image indices for {image_index} images.")

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Test accuracy of TinyCLIP on dataset')
    parser.add_argument('--imagenet_path', type=str, default='/network/scratch/s/sonia.joseph/datasets/kaggle_datasets', help='Path to dataset')
    parser.add_argument('--train_val', type=str, default='val', help='Train, test or validation set')
    parser.add_argument('--model_name', type=str, default='open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K', help='Model name to test')
    parser.add_argument('--save_dir', type=str, default='/network/scratch/s/sonia.joseph/CLIP_AUDIT/CLIP-ViT-B-32-DataComp.XL-s13B-b90K', help='Directory to save results')
    # parser.add_argument('--hook_point_name', type=str, default="blocks.{layer}.mlp.hook_post", help='Name of the activation to save')

    args = parser.parse_args()
    
    main(args.imagenet_path, args.train_val, args.model_name, args.save_dir)