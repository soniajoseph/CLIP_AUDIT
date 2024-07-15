# import h5py
# import numpy as np

# from clip_audit.utils.load_imagenet import load_imagenet, get_imagenet_names

# import torch

# from tqdm.auto import tqdm

# # import transforms
# from torchvision import transforms

# import matplotlib
# import matplotlib.pyplot as plt

# from vit_prisma.models.base_vit import HookedViT

# # import einops
# import einops

# from typing import List, Tuple

# import pandas as pd

# import os



# def get_activation_intervals(df: pd.DataFrame, layer: str, neuron: int) -> List[Tuple[float, float, float, float]]:
#     intervals = []
    

#     # Filter the DataFrame for the specific layer and neuron
#     row = df[(df['layer'] == layer) & (df['neuron'] == neuron)]
    
#     if not row.empty:
#         row = row.iloc[0]  # Get the first (and should be only) matching row
        
#         for i in range(-40, 41):
#             sd_min_key = f'sd_{i}_{i+1}_min'
#             sd_max_key = f'sd_{i}_{i+1}_max'
#             act_min_key = f'activation_{i}_{i+1}_min'
#             act_max_key = f'activation_{i}_{i+1}_max'
            
#             if i == -40:
#                 sd_min_key = 'sd_-inf_-40_min'
#                 sd_max_key = 'sd_-inf_-40_max'
#                 act_min_key = 'activation_-inf_-40_min'
#                 act_max_key = 'activation_-inf_-40_max'
#             elif i == 40:
#                 sd_min_key = 'sd_40_inf_min'
#                 sd_max_key = 'sd_40_inf_max'
#                 act_min_key = 'activation_40_inf_min'
#                 act_max_key = 'activation_40_inf_max'
            
#             sd_min = row.get(sd_min_key)
#             sd_max = row.get(sd_max_key)
#             act_min = row.get(act_min_key)
#             act_max = row.get(act_max_key)
            
#             if pd.notna(sd_min) and pd.notna(sd_max) and pd.notna(act_min) and pd.notna(act_max):
#                 intervals.append((
#                     float(sd_min),
#                     float(sd_max),
#                     float(act_min),
#                     float(act_max)
#                 ))
    
#     return intervals


# def sample_image_indices(file_path, activation_range, layer, neuron, n_samples=20):
#     """
#     Sample image indices for a given activation range, layer, and neuron.

#     Parameters:
#     - file_path (str): Path to the H5 file containing the activations
#     - activation_range (tuple): (min_activation, max_activation)
#     - layer (int): The layer number
#     - neuron (int): The neuron number
#     - n_samples (int): Number of samples to return (default 20)

#     Returns:
#     - list: Randomly sampled image indices
#     """
#     with h5py.File(file_path, 'r') as f:
#         # Get the activation data for the specified layer
#         layer_key = f'blocks.{layer}.hook_mlp_out'
#         activations = f[layer_key][:, :, neuron]  # Shape: (n_images, sequence_length)

#         # # Get the maximum activation for each image
#         # max_activations = np.max(activations, axis=1)

#         # Find indices where activations are within the specified range
#         min_act, max_act = activation_range
#         valid_indices = np.where((activations >= min_act) & (activations <= max_act))[0]

#         # If we have fewer valid indices than requested samples, return all of them
#         if len(valid_indices) <= n_samples:
#             sampled_indices = valid_indices
#         else:
#             # Randomly sample from the valid indices
#             sampled_indices = np.random.choice(valid_indices, size=n_samples, replace=False)

#         # Get the corresponding image indices
#         image_indices = f['image_indices'][:]
#         sampled_image_indices = image_indices[sampled_indices]

#     return sampled_image_indices.tolist()


# # get heatmaps of the top 20 images

# torch.no_grad()
# device = 'cuda'

# def get_all_activations(
          
#           image,
#           model,
#           hook_point_name,
#           neuron_idx
# ): 
#     image = image.to(device)
#     _, cache = model.run_with_cache(image.unsqueeze(0), names_filter=[hook_point_name])

#     post_reshaped = einops.rearrange(cache[hook_point_name], "batch seq d_mlp -> (batch seq) d_mlp")
#     post_reshaped = post_reshaped[:,neuron_idx]
#     # Compute activations (not from a fwd pass, but explicitly, by taking only the feature we want)
#     # This code is copied from the first part of the 'forward' method of the AutoEncoder class
#     # # sae_in =  post_reshaped - sparse_autoencoder.b_dec # Remove decoder bias as per Anthropic
#     # acts = einops.einsum(
#     #         sae_in,
#     #         sparse_autoencoder.W_enc[:, feature_id],
#     #         "x d_in, d_in -> x",
#     #     )
#     return post_reshaped 
     
# def image_patch_heatmap(activation_values,image_size=224, pixel_num=7):
#     activation_values = activation_values.detach().cpu().numpy()
#     activation_values = activation_values[1:]
#     activation_values = activation_values.reshape(pixel_num, pixel_num)

#     # Create a heatmap overlay
#     heatmap = np.zeros((image_size, image_size))
#     patch_size = image_size // pixel_num

#     for i in range(pixel_num):
#         for j in range(pixel_num):
#             heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = activation_values[i, j]

#     return heatmap

# def load_dataset(imagenet_path):
#     transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 # transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
#             ])
#     dataloader = load_imagenet(imagenet_path, 'val', shuffle=False, transform=transform)
#     dataset = dataloader.dataset
#     return dataset


# # Load files    
# hook_point_name = 'blocks.{layer}.hook_mlp_out'
# file_path = f'/network/scratch/s/sonia.joseph/CLIP_AUDIT/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/val/{hook_point_name}.h5'
# imagenet_path = '/network/scratch/s/sonia.joseph/datasets/kaggle_datasets'
# save_dir = '/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/TinyCLIP40M'

# # Load dataset
# dataset = load_dataset(imagenet_path)

# # Load model
# model_name = 'wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M'
# model = HookedViT.from_pretrained(model_name, is_clip=True, is_timm=False)
# model.to(device)

# # Load dataframe of activation intervals
# df_intervals_path = "/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/histograms/mlp.hook_out/all_neuron_activations_SD_intervals.csv"
# df_intervals = pd.read_csv(df_intervals_path)

# # Load neuron indices
# path = '/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/tinyclip_neuron_indices_mlp_out.npy'
# neuron_indices = np.load(path, allow_pickle=True).item()
# for layer_idx in tqdm(neuron_indices, desc="Layer"):
#     for neuron_idx in neuron_indices[layer_idx]:
#         layer_name = f'blocks.{layer_idx}.hook_mlp_out'

#         # Get neuron intervals based on standard deviatoin
#         print(f"Getting intervals for neuron {neuron_idx}, layer {layer_name}")
#         intervals = get_activation_intervals(df_intervals, layer=str(layer_name), neuron=neuron_idx)
#         for interval in intervals:

#             sd_min, sd_max, activation_min, activation_max = interval

#             # print(f"Sampling images for neuron {neuron_idx}, layer {layer_idx}, activation range {interval}")

#             sampled_indices = sample_image_indices(file_path, (activation_min, activation_max), layer_idx, neuron_idx)

#             # if indices are less than 20, skip interval
#             if len(sampled_indices) < 20:
#                 print(f"Skipping interval {interval} as it has less than 20 images")
#                 continue

#             images = []
#             for idx in sampled_indices:
#                 image, label, _ = dataset[idx]
#                 images.append(image)

                
#             grid_size = int(np.ceil(np.sqrt(len(images))))
#             fig, axs = plt.subplots(int(np.ceil(len(images)/grid_size)), grid_size, figsize=(15, 15))

#             fig.suptitle(f"Neuron {neuron_idx}, Layer {layer_idx}, SD range: {sd_min} - {sd_max}, Activation range: {activation_min} - {activation_max}")
#             for ax in axs.flatten():
#                 ax.axis('off')
        
#             for idx, image in enumerate(images):

#                 row = idx // grid_size
#                 col = idx % grid_size

#                 all_activations = get_all_activations(image, model, hook_point_name=f'blocks.{layer_idx}.hook_mlp_out', neuron_idx=neuron_idx)
#                 heatmap = image_patch_heatmap(all_activations)

#                 axs[row, col].imshow(image.permute(1, 2, 0))
#                 axs[row, col].imshow(heatmap, cmap='viridis', alpha=0.3)  # Overlaying the heatmap
#                 # axs[row, col].set_title(f"{label} {val.item():0.03f} {'class token!' if has_zero else ''}")  
#                 axs[row, col].axis('off')  


#             save_name = f"{save_dir}/layer_{layer_idx}/neuron_{neuron_idx}/neuron_{neuron_idx}_layer_{layer_idx}_activation_{interval}.png"
#             # create directory if it doesn't exist
#             os.makedirs(os.path.dirname(save_name), exist_ok=True)
#             plt.tight_layout()
#             plt.savefig(save_name)
#             plt.close()

# # main functio

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from vit_prisma.models.base_vit import HookedViT
import einops
from typing import List, Tuple
import pandas as pd
import os
from clip_audit.utils.load_imagenet import load_imagenet


def get_top_activations_only(file_path, output_dir, neuron, layer_idx, df, n_samples=20):
    with h5py.File(file_path, 'r') as f:
        layer_key = f'blocks.{layer_idx}.hook_mlp_out'
        activations = f[layer_key][:, :, neuron]

        # Get indices for max activations across all tokens
        max_activations = np.max(activations, axis=1)  # Max activation per image
        top_k_indices = np.argsort(max_activations)[-n_samples:][::-1]  # Top k indices
        # top k activation values
        top_k_activations = max_activations[top_k_indices]
        
        image_indices = f['image_indices'][:]
        sampled_image_indices = image_indices[top_k_indices]

        # calculate standard deviaion from file
        sd_interval = get_sd_interval_for_activation(df, layer_name=layer_key, layer=layer_idx, neuron=neuron, act_min=np.min(top_k_activations), act_max=np.max(top_k_activations))

    
    return sampled_image_indices.tolist(), top_k_activations, sd_interval

def get_sd_interval_for_activation(df: pd.DataFrame, layer_name: str, layer: int, neuron: int, act_min: float, act_max: float) -> Tuple[float, float]:
    rows = df[(df['layer'] == layer_name)  & (df['neuron'] == neuron)]
    
    if rows.empty:
        return None  # Return None if no matching rows are found
        
    sd_min_global = float('inf')
    sd_max_global = float('-inf')
    found_interval = False

    for _, row in rows.iterrows():
        for i in range(-40, 41):
            sd_min_key, sd_max_key, act_min_key, act_max_key = get_interval_keys(i)
            
            current_act_min = row.get(act_min_key)
            current_act_max = row.get(act_max_key)
            
            if pd.notna(current_act_min) and pd.notna(current_act_max):
                current_act_min = float(current_act_min)
                current_act_max = float(current_act_max)
                
                # Check if there's any overlap between the intervals
                if not (current_act_max < act_min or act_max < current_act_min):
                    sd_min = row.get(sd_min_key)
                    sd_max = row.get(sd_max_key)
                    
                    if pd.notna(sd_min) and pd.notna(sd_max):
                        sd_min_global = min(sd_min_global, float(sd_min))
                        sd_max_global = max(sd_max_global, float(sd_max))
                        found_interval = True

    if found_interval:
        return sd_min_global, sd_max_global
    
    return None
    


def get_activation_intervals(df: pd.DataFrame, layer: str, neuron: int) -> List[Tuple[float, float, float, float]]:
    intervals = []
    row = df[(df['layer'] == layer) & (df['neuron'] == neuron)]
    
    if not row.empty:
        row = row.iloc[0]
        for i in range(-40, 41):
            sd_min_key, sd_max_key, act_min_key, act_max_key = get_interval_keys(i)
            
            sd_min, sd_max = row.get(sd_min_key), row.get(sd_max_key)
            act_min, act_max = row.get(act_min_key), row.get(act_max_key)
            
            if all(pd.notna(x) for x in [sd_min, sd_max, act_min, act_max]):
                intervals.append((float(sd_min), float(sd_max), float(act_min), float(act_max)))
    
    return intervals

def get_interval_keys(i):
    if i == -40:
        return 'sd_-inf_-40_min', 'sd_-inf_-40_max', 'activation_-inf_-40_min', 'activation_-inf_-40_max'
    elif i == 40:
        return 'sd_40_inf_min', 'sd_40_inf_max', 'activation_40_inf_min', 'activation_40_inf_max'
    else:
        return f'sd_{i}_{i+1}_min', f'sd_{i}_{i+1}_max', f'activation_{i}_{i+1}_min', f'activation_{i}_{i+1}_max'

def sample_unique_image_indices(file_path, activation_range, layer, neuron, n_samples=20):
    with h5py.File(file_path, 'r') as f:
        layer_key = f'blocks.{layer}.hook_mlp_out'
        activations = f[layer_key][:, :, neuron]
        min_act, max_act = activation_range
        
        # Find images where any patch falls within the activation range
        valid_image_mask = np.any((activations >= min_act) & (activations <= max_act), axis=1)
        valid_image_indices = np.where(valid_image_mask)[0]
        
        # Sample from these unique image indices
        if len(valid_image_indices) <= n_samples:
            sampled_indices = valid_image_indices
        else:
            sampled_indices = np.random.choice(valid_image_indices, size=n_samples, replace=False)
        
        image_indices = f['image_indices'][:]
        sampled_image_indices = image_indices[sampled_indices]
    
    return sampled_image_indices.tolist()
        
    
    return sampled_image_indices.tolist()

def get_all_activations(image, model, hook_point_name, neuron_idx):
    device = 'cuda'
    with torch.no_grad():
        image = image.to(device)
        _, cache = model.run_with_cache(image.unsqueeze(0), names_filter=[hook_point_name])
        post_reshaped = einops.rearrange(cache[hook_point_name], "batch seq d_mlp -> (batch seq) d_mlp")
        return post_reshaped[:, neuron_idx]

def image_patch_heatmap(activation_values, image_size=224, pixel_num=7):
    activation_values = activation_values.detach().cpu().numpy()[1:]
    activation_values = activation_values.reshape(pixel_num, pixel_num)
    heatmap = np.zeros((image_size, image_size))
    patch_size = image_size // pixel_num
    for i in range(pixel_num):
        for j in range(pixel_num):
            heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = activation_values[i, j]
    return heatmap

def load_dataset(imagenet_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataloader = load_imagenet(imagenet_path, 'val', shuffle=False, transform=transform)
    return dataloader.dataset

def process_neuron(layer_idx, neuron_idx, model, dataset, df_intervals, file_path, save_dir, hook_point_name, top_k_only=False):

    layer_name = f'blocks.{layer_idx}.hook_mlp_out'

    if not top_k_only:

        intervals = get_activation_intervals(df_intervals, layer=str(layer_name), neuron=neuron_idx)
        
        for interval in intervals:
            sd_min, sd_max, activation_min, activation_max = interval
            sampled_indices = sample_image_indices(file_path, (activation_min, activation_max), layer_idx, neuron_idx)
            
            if len(sampled_indices) < 20:
                print(f"Skipping interval {interval} as it has less than 20 images")
                continue
            
            images = [dataset[idx][0] for idx in sampled_indices]
            plot_images(images, layer_idx, neuron_idx, sd_min, sd_max, activation_min, activation_max, model, hook_point_name, save_dir)
    
    else:
        sampled_indices, top_k_activations, sd_intervals = get_top_activations_only(file_path, save_dir, neuron_idx, layer_idx, df_intervals)
        print(f"Neuron {neuron_idx}, Layer {layer_idx}, SD intervals: {sd_intervals}")
        images = [dataset[idx][0] for idx in sampled_indices]
        # get min and max activation
        activation_min, activation_max = np.min(top_k_activations), np.max(top_k_activations)
        plot_images(images, layer_idx, neuron_idx, sd_intervals[0], sd_intervals[1], activation_min, activation_max, model, hook_point_name, save_dir, top_k_only=True)

def plot_images(images, layer_idx, neuron_idx, sd_min, sd_max, activation_min, activation_max, model, hook_point_name, save_dir, top_k_only=False):
    grid_size = int(np.ceil(np.sqrt(len(images))))
    fig, axs = plt.subplots(int(np.ceil(len(images)/grid_size)), grid_size, figsize=(15, 15))
    fig.suptitle(f"Neuron {neuron_idx}, Layer {layer_idx}, SD range: {sd_min} - {sd_max}, Activation range: {activation_min} - {activation_max}")
    
    for ax in axs.flatten():
        ax.axis('off')
    
    for idx, image in enumerate(images):
        row, col = divmod(idx, grid_size)
        all_activations = get_all_activations(image, model, hook_point_name=f'blocks.{layer_idx}.hook_mlp_out', neuron_idx=neuron_idx)
        heatmap = image_patch_heatmap(all_activations)
        axs[row, col].imshow(image.permute(1, 2, 0))
        axs[row, col].imshow(heatmap, cmap='viridis', alpha=0.3)
        axs[row, col].axis('off')
    
    if top_k_only:
        save_name = f"{save_dir}/layer_{layer_idx}/neuron_{neuron_idx}/neuron_{neuron_idx}_layer_{layer_idx}_top_k_only.png"
    else:
        save_name = f"{save_dir}/layer_{layer_idx}/neuron_{neuron_idx}/neuron_{neuron_idx}_layer_{layer_idx}_activation_{sd_min}_{sd_max}_{activation_min}_{activation_max}.png"
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_name)
    print(f"Saved image to {save_name}")
    plt.close()

def main(args):
    hook_point_name = 'blocks.{layer}.hook_mlp_out'
    file_path = f'/network/scratch/s/sonia.joseph/CLIP_AUDIT/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/val/{hook_point_name}.h5'
    imagenet_path = '/network/scratch/s/sonia.joseph/datasets/kaggle_datasets'
    save_dir = '/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/TinyCLIP40M'
    df_intervals_path = "/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/histograms/mlp.hook_out/all_neuron_activations_SD_intervals.csv"
    neuron_indices_path = '/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/tinyclip_neuron_indices_mlp_out.npy'

    dataset = load_dataset(imagenet_path)
    model_name = 'wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M'
    model = HookedViT.from_pretrained(model_name, is_clip=True, is_timm=False).to('cuda')
    df_intervals = pd.read_csv(df_intervals_path)
    neuron_indices = np.load(neuron_indices_path, allow_pickle=True).item()


    if args.layer_idx is not None:
        if args.layer_idx in neuron_indices:
            layer_indices = [args.layer_idx]
        else:
            print(f"Layer {args.layer_idx} not found in neuron indices. Exiting.")
            return
    else:
        layer_indices = neuron_indices.keys()

    for layer_idx in tqdm(layer_indices, desc="Layer"):
        for neuron_idx in tqdm(neuron_indices[layer_idx], desc="Neuron"):
            save_name = f"{save_dir}/layer_{layer_idx}/neuron_{neuron_idx}/"
            # if directory exists, skip this neuron
            if os.path.exists(save_name) and not args.top_k_only:
                print(f"Neuron {neuron_idx} in layer {layer_idx} already processed. Skipping.")
                continue
            process_neuron(layer_idx, neuron_idx, model, dataset, df_intervals, file_path, save_dir, hook_point_name, args.top_k_only)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process neurons in specific layers of a neural network.")
    parser.add_argument("--layer_idx", type=int, help="Specific layer index to process. If not provided, all layers will be processed.", default=None)
    # get top k only, store true, use as a boolean
    parser.add_argument("--top_k_only", action="store_true", help="Get only the top k activations for each neuron.")
    args = parser.parse_args()
    main(args)