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

import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

from PIL import Image

from torchvision.transforms.functional import InterpolationMode


from vit_prisma.transforms.open_clip_transforms import get_clip_val_transforms 
from vit_prisma.dataloaders.imagenet_classes_simple import imagenet_classes

from clip_audit.dataloader.conceptual_captions import load_conceptual_captions

VERBOSE = True
SAVE_FIGURES = True

def print_available_layers(file_path):
    with h5py.File(file_path, 'r') as f:
        print("Available layers in the file:")
        for key in f.keys():
            if key != 'image_indices':
                print(f"- {key}")

def get_extreme_activations(file_path, neuron_idx, layer_idx, layer_type, n_samples=20, type_of_sampling='avg'):
    # print_available_layers(file_path)
    file_path = os.path.join(file_path, f"{layer_type}.h5")
    # print(f"File path: {file_path}")
    # print_available_layers(file_path)
    with h5py.File(file_path, 'r') as f:
        # print("All keys:", list(f.keys()))

        if layer_type == 'hook_post':
            layer_key = f'blocks.{layer_idx}.mlp.hook_post'
        else:
            layer_key = f'blocks.{layer_idx}.{layer_type}'
        if layer_key not in f:
            print(f"Warning: Layer key '{layer_key}' not found in the file.")
            return None, None, None, None

        activations = f[layer_key][:, :, neuron_idx]

        # print(f"Activations shape for {layer_key}", activations.shape)
        # print(f"Type of sampling: {type_of_sampling}")

        if type_of_sampling == 'max':
            aggregated_activations = np.max(activations, axis=1)
        elif type_of_sampling == 'avg':
            aggregated_activations = np.mean(activations, axis=1)
        elif type_of_sampling == 'max_cls':
            aggregated_activations = activations[:, 0]
        else:
            raise ValueError(f"Unknown sampling type: {type_of_sampling}")

        sorted_indices = np.argsort(aggregated_activations)
        bottom_k_indices = sorted_indices[:n_samples]
        top_k_indices = sorted_indices[-n_samples:]
        
        top_k_activations = aggregated_activations[top_k_indices]
        bottom_k_activations = aggregated_activations[bottom_k_indices]
        
        image_indices = f['image_indices'][:]
        top_sampled_image_indices = image_indices[top_k_indices]
        bottom_sampled_image_indices = image_indices[bottom_k_indices]

        # print(f"Top sampled image indices for {layer_key}: {top_sampled_image_indices}")
        # print(f"Bottom sampled image indices for {layer_key}: {bottom_sampled_image_indices}")
    
    return top_sampled_image_indices.tolist(), top_k_activations, bottom_sampled_image_indices.tolist(), bottom_k_activations

def process_neuron(layer_idx, neuron_idx, model, dataset, file_path, save_dir, dataset_name, type_of_sampling='max', imagenet_classes=None):
    
    layer_types = ["hook_mlp_out", "hook_resid_post"]

    print(f"Processing neuron {neuron_idx} in layer {layer_idx}") if VERBOSE else None

    

    for layer_type in layer_types:
        top_indices, top_activations, bottom_indices, bottom_activations = get_extreme_activations(file_path, neuron_idx, layer_idx, layer_type, type_of_sampling=type_of_sampling)
        # print(f"Neuron {neuron_idx}, Layer {layer_idx}, Layer Type {layer_type}")

        if dataset_name == 'imagenet':
            image_key = 0
            label_key = 1
        elif dataset_name == 'conceptual_captions':
            image_key = 'image'
            label_key = 'caption'

        top_images = [dataset[idx][image_key] for idx in top_indices]
        bottom_images = [dataset[idx][image_key] for idx in bottom_indices]

        if dataset_name == 'imagenet':
            top_class_names = [imagenet_classes[dataset[idx][label_key]] for idx in top_indices]
            bottom_class_names = [imagenet_classes[dataset[idx][label_key]] for idx in bottom_indices]
        elif dataset_name == 'conceptual_captions':
            # top_class_names = [dataset[idx][label_key] for idx in top_indices]
            # bottom_class_names = [dataset[idx][label_key] for idx in bottom_indices]
            top_class_names = ['' for idx in top_indices]
            bottom_class_names = ['' for idx in bottom_indices]

        # Save values and indices
        top_dir = f"{save_dir}/layer_{layer_idx}/neuron_{neuron_idx}/{layer_type}/{type_of_sampling}/top"
        bot_dir = f"{save_dir}/layer_{layer_idx}/neuron_{neuron_idx}/{layer_type}/{type_of_sampling}/bottom"

        # if directory doesn't exist, make it
        os.makedirs(top_dir, exist_ok=True)
        os.makedirs(bot_dir, exist_ok=True)

        # Save indices and activations
        torch.save({
            'indices': top_indices,
            'activations': top_activations
        }, f"{top_dir}/indices_and_activations.pt")
        
        torch.save({
            'indices': bottom_indices,
            'activations': bottom_activations
        }, f"{bot_dir}/indices_and_activations.pt")

        # Plot figures
        if SAVE_FIGURES:
            plot_images(top_images, top_indices, top_class_names, layer_idx, neuron_idx, model, save_dir, type_of_sampling=type_of_sampling, activations=top_activations, extreme_type='top', layer_type=layer_type, file_path=file_path)
            plot_images(bottom_images, bottom_indices, bottom_class_names, layer_idx, neuron_idx, model, save_dir, type_of_sampling=type_of_sampling, activations=bottom_activations, extreme_type='bottom', layer_type=layer_type, file_path=file_path)

def tensor_to_pil(tensor):
    """Convert a tensor to a PIL Image."""
    return Image.fromarray((tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))



def process_display_image(image):
    # Convert to numpy array if it isn't already
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure the values are in 0-255 range
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    return image

def get_clip_val_transforms_display(image_size=224):
    """
    Transform for display that preserves the visual quality of the image
    """
    return transforms.Compose([
        transforms.Resize(size=image_size, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(size=(image_size, image_size)),
        # No need for ToTensor() and permute since we want to keep the image displayable
    ])

def plot_images(images, image_indices, class_names, layer_idx, neuron_idx, model, save_dir, type_of_sampling='max', activations=None, extreme_type='top', layer_type=None, file_path=None, save_figures=SAVE_FIGURES):
    grid_size = int(np.ceil(np.sqrt(len(images))))
    
    def create_figure(include_heatmap=True):
        fig, axs = plt.subplots(int(np.ceil(len(images)/grid_size)), grid_size, figsize=(20, 20))
        fig.suptitle(f"Neuron {neuron_idx}, Layer {layer_idx}, {extreme_type.capitalize()} Activations ({layer_type}, {type_of_sampling})", fontsize=16)
        
        axs = axs.flatten()
        for ax in axs:
            ax.axis('off')

        display_transform = get_clip_val_transforms_display()
        full_transform = get_clip_val_transforms()


        if layer_type == 'hook_post':
            key = f"blocks.{layer_idx}.mlp.hook_post"
        else:
            key = f'blocks.{layer_idx}.{layer_type}'


        for idx, (image, image_idx, class_name) in enumerate(zip(images, image_indices, class_names)):

            # img_pil = transforms.ToPILImage()(image)
            # img_pil = transforms.Resize(224, interpolation=InterpolationMode.BICUBIC)(img_pil)
            # img_pil = transforms.CenterCrop(224)(img_pil)
            # img_pil = transforms.CenterCrop(224)(img_pil)
            if isinstance(image, Image.Image):
                display_image = display_transform(image)
            else:
                img = image.permute(1, 2, 0).numpy()
                display_image = (img - img.min()) / (img.max() - img.min())
                
            ax = axs[idx]

            if include_heatmap:
                ax.imshow(display_image)

                # Heatmap
                all_activations = get_all_activations(image_idx, layer_idx, layer_type, neuron_idx, file_path)
                pixel_num = int(np.sqrt(all_activations.shape[0]-1))
                heatmap = image_patch_heatmap(all_activations, pixel_num=pixel_num)
                ax.imshow(heatmap, cmap='viridis', alpha=0.3)
            else:
                ax.imshow(display_image)
            
            ax.axis('off')
                    
            fontsize = 14
            if activations is not None:
                ax.set_title(f"{class_name}  {activations[idx]:.4f}", fontsize=fontsize)
            else:
                ax.set_title(f"{class_name}", fontsize=fontsize)

        return fig, axs

    # Create figures
    fig_with_heatmap, _ = create_figure(include_heatmap=True)
    fig_without_heatmap, _ = create_figure(include_heatmap=False)

    # Prepare file names and directories
    base_name = f"neuron_{neuron_idx}_layer_{layer_idx}_{extreme_type}_{type_of_sampling}_{layer_type}"
    
    # Function to save figures
    def save_figure(fig, name_suffix, is_svg=False):
        if is_svg:
            base_dir = f"{save_dir}/svg/layer_{layer_idx}/neuron_{neuron_idx}/{layer_type}/{type_of_sampling}/{extreme_type}"
            file_ext = "svg"
        else:
            base_dir = f"{save_dir}/layer_{layer_idx}/neuron_{neuron_idx}/{layer_type}/{type_of_sampling}/{extreme_type}"
            file_ext = "png"
        
        os.makedirs(base_dir, exist_ok=True)
        save_path = f"{base_dir}/{base_name}{name_suffix}.{file_ext}"
        
        fig.tight_layout()
        if is_svg:
            fig.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {file_ext.upper()} image to {save_path}") if VERBOSE else None

    # Save figures
    if save_figures:
        save_figure(fig_with_heatmap, "", is_svg=False)
        save_figure(fig_with_heatmap, "", is_svg=True)
        save_figure(fig_without_heatmap, "_no_heatmap", is_svg=False)
        save_figure(fig_without_heatmap, "_no_heatmap", is_svg=True)
    
    


    # # Save individual images
    # img_save_dir = f"{save_dir}/layer_{layer_idx}/neuron_{neuron_idx}/{layer_type}/{sampling_type}/{extreme_type}/individual_images"
    # os.makedirs(img_save_dir, exist_ok=True)
    # display_transform = get_clip_val_transforms_display()
    # for idx, image in enumerate(images):
    #     display_image = display_transform(image)
    #     img_save_path = f"{img_save_dir}/image_{idx}.png"
    #     if isinstance(display_image, Image.Image):
    #         display_image.save(img_save_path)
    #     else:
    #         Image.fromarray((display_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(img_save_path)


    plt.close('all')



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

# def sample_unique_image_indices(file_path, activation_range, layer, neuron, layer_type, n_samples=20):
#     with h5py.File(file_path, 'r') as f:
#         layer_key = f'blocks.{layer}.{layer_type}'
#         activations = f[layer_key][:, :, neuron]
#         min_act, max_act = activation_range
        
#         # Find images where any patch falls within the activation range
#         valid_image_mask = np.any((activations >= min_act) & (activations <= max_act), axis=1)
#         valid_image_indices = np.where(valid_image_mask)[0]
        
#         # Sample from these unique image indices
#         if len(valid_image_indices) <= n_samples:
#             sampled_indices = valid_image_indices
#         else:
#             sampled_indices = np.random.choice(valid_image_indices, size=n_samples, replace=False)
        
#         image_indices = f['image_indices'][:]
#         sampled_image_indices = image_indices[sampled_indices]
    
#     return sampled_image_indices.tolist()
        
    
    return sampled_image_indices.tolist()

# def get_all_activations(image, model, hook_point_name, neuron_idx):
#     device = 'cuda'
#     model.eval()
#     with torch.no_grad():
#         image = image.to(device)
#         _, cache = model.run_with_cache(image.unsqueeze(0), names_filter=[hook_point_name])
#         # print(f"Cache shape: {cache[hook_point_name].shape}")
#         post_reshaped = einops.rearrange(cache[hook_point_name], "batch seq d_mlp -> (batch seq) d_mlp")
#         # print(f"Post reshaped shape: {post_reshaped.shape}")
#         return post_reshaped[:, neuron_idx]

def get_all_activations(image_index, layer_idx, layer_type, neuron_idx, file_path):
    file_path = os.path.join(file_path, f"{layer_type}.h5")
    with h5py.File(file_path, 'r') as f:

        if layer_type == 'hook_post':
            layer_key = f'blocks.{layer_idx}.mlp.hook_post'
        else:
            layer_key = f'blocks.{layer_idx}.{layer_type}'
        
        activations = f[layer_key][image_index, :, neuron_idx]
    
    return activations

def image_patch_heatmap(activation_values, image_size=224, pixel_num=7):
    # activation_values = activation_values.detach().cpu().numpy()[1:]
    activation_values = activation_values[1:]
    activation_values = activation_values.reshape(pixel_num, pixel_num)
    heatmap = np.zeros((image_size, image_size))
    patch_size = image_size // pixel_num
    for i in range(pixel_num):
        for j in range(pixel_num):
            heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = activation_values[i, j]
    return heatmap

def load_dataset(imagenet_path,train_or_test):
    # transform = get_clip_val_transforms()
    
    dataloader = load_imagenet(imagenet_path, train_or_test, shuffle=False, transform=None)
    return dataloader.dataset

def create_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Process neurons in specific layers of a neural network.")
    
    # Model and path configurations
    parser.add_argument("--model_name", 
                       default='open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K',
                    # default = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                       help="Name of the model to use")
    
    parser.add_argument("--dataset_name", 
                       default='conceptual_captions',)

    parser.add_argument("--train_or_test",
                       default='train',)
    
    parser.add_argument("--imagenet_path",
                       default='/network/scratch/s/sonia.joseph/datasets/kaggle_datasets',
                       help="Path to ImageNet dataset")
    
    parser.add_argument("--neuron_indices_path",
                       default='/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/clip_base_mlp_out.npy',
                       help="Path to neuron indices file")
    
    # Processing options
    parser.add_argument("--layer_idx", 
                       type=int,
                       default=None,
                       help="Specific layer index to process. If not provided, all layers will be processed.")
    
    parser.add_argument("--replace",
                       action="store_true",
                       help="Rerun despite folder already being there")
    
    parser.add_argument("--type_of_sampling",
                       type=str,
                       default='avg',
                       help="Type of sampling to use for selecting top k activations")
    
    parser.add_argument("--verbose",
                       action="store_true",
                       help="Print verbose output")
    
    parser.add_argument("--all_neurons",
                       action="store_true",
                       help="Do every neuron, not just randomly sampled ones")

    return parser


def main(args):
    # model_name = 'open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K'
    # file_path = f'/network/scratch/s/sonia.joseph/CLIP_AUDIT/CLIP-ViT-B-32-DataComp.XL-s13B-b90K/open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K/val/'
    # imagenet_path = '/network/scratch/s/sonia.joseph/datasets/kaggle_datasets'
    # save_dir = f'/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/CLIP-ViT-B-32-DataComp.XL-s13B-b90K/'
    # # df_intervals_path = "/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/histograms/mlp.hook_out/all_neuron_activations_SD_intervals.csv"
    # neuron_indices_path = '/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/clip_base_mlp_out.npy'

    clean_model_name = args.model_name.replace("/", "_")
    clean_model_name = clean_model_name.replace(":", "_")
    file_path = f'/network/scratch/s/sonia.joseph/CLIP_AUDIT/{clean_model_name}/{args.dataset_name}/{args.train_or_test}'
    save_dir = f'/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/{clean_model_name}/{args.dataset_name}/{args.train_or_test}'

    
    if args.dataset_name == 'imagenet':
        dataset = load_dataset(args.imagenet_path, args.train_or_test)
    elif args.dataset_name == 'conceptual_captions':
        dataset = load_conceptual_captions(args.train_or_test, dataloader=False)

    model = HookedViT.from_pretrained(args.model_name, is_clip=True, is_timm=False, fold_ln=False).to('cuda')
    # df_intervals = pd.read_csv(df_intervals_path)

    print("Number of layers:", model.cfg.n_layers)
    if args.all_neurons:
        if model.cfg.n_layers > 12:
            range_object = range(0, model.cfg.n_layers, 4)
        else:
            range_object = range(model.cfg.n_layers)
        neuron_indices = {layer_idx: np.arange(30) for layer_idx in range_object}
        save_dir = f'{save_dir}/all_neurons'
        # make dir if doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    else:
        neuron_indices = np.load(args.neuron_indices_path, allow_pickle=True).item()
    
    print("Neuron indices:", neuron_indices)

    if args.layer_idx is not None:
       layer_indices = [args.layer_idx]
    else:
        layer_indices = neuron_indices.keys()

    for layer_idx in tqdm(layer_indices, desc="Layer"):
        for neuron_idx in tqdm(neuron_indices[layer_idx], desc="Neuron"):
            save_name = f"{save_dir}/layer_{layer_idx}/neuron_{neuron_idx}/"
            # if directory exists, skip this neuron
            if os.path.exists(save_name) and not args.replace:
                print(f"Neuron {neuron_idx} in layer {layer_idx} already processed. Skipping.")
                continue
            process_neuron(layer_idx, neuron_idx, model, dataset, file_path, save_dir, args.dataset_name, args.type_of_sampling, imagenet_classes)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
    
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Process neurons in specific layers of a neural network.")
#     parser.add_argument("--layer_idx", type=int, help="Specific layer index to process. If not provided, all layers will be processed.", default=None)
#     # get top k only, store true, use as a boolean
#     parser.add_argument("--replace", action="store_true", help="Rerun despite folder already being there.")
#     parser.add_argument("--type_of_sampling", type=str, default='avg', help="Type of sampling to use for selecting top k activations.")
#     parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
#     parser.add_argument("--all_neurons", action="store_true", help="Do every neuron, not just randomly sampled ones.")
#     args = parser.parse_args()
#     main(args)
