
"""
Test accuracy of TinyCLIP on dataset.

Mainly as sanity check and also making sure we're doing our image normalization properly
"""

import os
from clip_audit.utils.load_imagenet import load_imagenet, get_imagenet_names

from clip_audit.dataloader.conceptual_captions import load_conceptual_captions


import argparse
from transformers import CLIPProcessor, CLIPModel
import torch

from vit_prisma.models.base_vit import HookedViT

from torchvision import transforms


from tqdm.auto import tqdm
import numpy as np
import h5py

from vit_prisma.transforms.open_clip_transforms import get_clip_val_transforms


def create_save_path(save_dir, model_name, dataset_name, train_val, layer_type):
    # clean model name of '/'
    model_name = model_name.replace('/', '_')
    model_name = model_name.replace(':', '_')
    
    # Construct full path
    full_path = os.path.join(save_dir, model_name, dataset_name, train_val, f"{layer_type}.h5")
    
    # Create all parent directories
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    return full_path

def main(imagenet_path, train_val, model_name, dataset_name, save_dir, neuron_indices=None):

    # Load data
    if dataset_name == 'imagenet':
        ind_to_name, imagenet_names = get_imagenet_names(imagenet_path)
        transforms = get_clip_val_transforms()
        dataloader = load_imagenet(imagenet_path, train_val, shuffle=False, transform=transforms)
    elif dataset_name == 'conceptual_captions':
        dataloader = load_conceptual_captions(train_val)
        
    if neuron_indices and model_name == 'open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K':
        neuron_indices_mlp_out = np.load('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/clip_base_mlp_out.npy', allow_pickle=True).item()
        neuron_indices_resid_post = np.load('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/clip_base_residual_post.npy', allow_pickle=True).item()
        
        # if neuron_indices and model_name == 'open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K':
        #     neuron_indices_mlp_out = np.load('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/vit_g_mlp_out.npy', allow_pickle=True).item()
        #     neuron_indices_resid_post = np.load('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/vit_g_resid.npy', allow_pickle=True).item()

        # Load model
    device = 'cuda'
    model = HookedViT.from_pretrained(model_name, is_timm=False, is_clip=True, fold_ln=False, center_writing_weights=False) # in future, do all models
    model.to(device)
    print(f"Model loaded: {model_name}")

    layer_num = model.cfg.n_layers
    print(f"Number of layers: {layer_num}")
    if layer_num > 12:
        stride = 4
        total_range = range(0, layer_num, stride)  # if doing vit-big, do every 4 layers, starting at layer 0 
    else:
        total_range = range(layer_num)

    

    # Create names_filter
    hook_point_names = []
    for layer in total_range:  
        hook_point_names.extend([
            f"blocks.{layer}.hook_mlp_out",
            f"blocks.{layer}.hook_resid_post",
            # f"blocks.{layer}.mlp.hook_post"
        ])
    names_filter = lambda name: name in hook_point_names

    layer_types = ["hook_mlp_out", "hook_resid_post"]
    files = {layer_type: h5py.File(create_save_path(save_dir, model_name, dataset_name, train_val, layer_type), 'w') 
             for layer_type in layer_types}

    image_index = 0
    with torch.no_grad():
        for output in tqdm(dataloader, desc="Evaluating"):
            
            if dataset_name == 'imagenet':
                images = output[0].to(device)
                # labels = labels.to(device)
            elif dataset_name == 'conceptual_captions':
                images = output['image'].to(device)

            output, cache = model.run_with_cache(images, names_filter=names_filter)

            # if cache is empty, show warning
            if not cache:
                print(f"Warning: cache is empty for image index {image_index}.")

            for layer, activation in cache.items():


                if neuron_indices:
                    layer_idx = int(layer.split('.')[1]) 
                    if 'hook_mlp_out' in layer:
                        selected_neurons = neuron_indices_mlp_out[layer_idx]
                        # print(selected_neurons)
                        # print(activation.shape)
                        activation = activation[:, :, selected_neurons]
                    elif 'hook_resid_post' in layer:
                        selected_neurons = neuron_indices_resid_post[layer_idx]
                        # print(selected_neurons)
                        # print(activation.shape)
                        activation = activation[:, :, selected_neurons]

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
    parser.add_argument('--train_val', type=str, default='train', help='Train, test or validation set')
    parser.add_argument('--model_name', type=str, default='open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K', help='Model name to test')
    parser.add_argument('--dataset_name', type=str, default='conceptual_captions', help='Dataset name')
    parser.add_argument('--save_dir', type=str, default='/network/scratch/s/sonia.joseph/CLIP_AUDIT/', help='Directory to save results')
    parser.add_argument('--neuron_indices', type=bool, default=False, help='Whether to save neuron indices')
    # parser.add_argument('--hook_point_name', type=str, default="blocks.{layer}.mlp.hook_post", help='Name of the activation to save')

    args = parser.parse_args()
    
    main(args.imagenet_path, args.train_val, args.model_name, args.dataset_name, args.save_dir, args.neuron_indices)