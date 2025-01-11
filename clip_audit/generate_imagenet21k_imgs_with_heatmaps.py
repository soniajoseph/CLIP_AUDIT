
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import h5py
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from vit_prisma.transforms.open_clip_transforms import get_clip_val_transforms 

from vit_prisma.models.base_vit import HookedViT

# Constants
SAVE_FIGURES = True
VERBOSE = True
IMAGE_SIZE = 224

def process_display_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    return image

def get_clip_val_transforms_display(image_size=224):
    return transforms.Compose([
        transforms.Resize(size=image_size, interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(size=(image_size, image_size)),
    ])



# def visualize_neuron_activations(checkpoint_path, save_dir, image_path, model):
#     checkpoint = torch.load(checkpoint_path)

#     # load neuron indices
#     neuron_indices_mlp_out = np.load('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/clip_base_mlp_out.npy', allow_pickle=True).item()
#     neuron_indices_resid_post = np.load('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/clip_base_residual_post.npy', allow_pickle=True).item()

    
#     for strategy in checkpoint.keys():
#         for interval in checkpoint[strategy].keys():
#             for layer_type in checkpoint[strategy][interval].keys():
#                 data = checkpoint[strategy][interval][layer_type]
                
#                 for layer_idx in tqdm(range(len(data['activations']))):
#                     for neuron_idx in range(data['activations'][layer_idx].shape[0])[:15]: # stop at 15 neurons
#                         # Get data for this neuron
#                         activations = data['activations'][layer_idx][neuron_idx]
#                         image_ids = data['image_ids'][layer_idx][neuron_idx]
                        
#                         # Load images
#                         images = []
#                         class_names = []  # You might want to load actual class names if available
#                         for img_id in image_ids:
#                             img_path = f"{image_path}/{img_id}.jpg"
#                             if os.path.exists(img_path):
#                                 img = Image.open(img_path)
#                                 images.append(img)
#                                 class_names.append(f"Image {img_id}")
                        
#                         # Plot and save images
#                         plot_images(
#                             images=images,
#                             image_indices=image_ids,
#                             class_names=class_names,
#                             layer_idx=layer_idx,
#                             neuron_idx=neuron_idx,
#                             model=model,  # Not needed for visualization
#                             save_dir=save_dir,
#                             type_of_sampling=strategy,
#                             activations=activations,
#                             extreme_type=interval,
#                             layer_type=layer_type,
#                             # file_path=activation_file_path,
#                             save_figures=SAVE_FIGURES,
#                             neuron_indices_mlp_out=neuron_indices_mlp_out,
#                             neuron_indices_resid_post=neuron_indices_resid_post,
#                         )


def get_imagenet21k_class_name(class_id):
    # If input is in format "Image nXXXXXXXX_YYYY", extract the ID
    if isinstance(class_id, str) and 'Image n' in class_id:
        # Extract just the 'nXXXXXXXX' part
        formatted_id = class_id.split('_')[0].replace('Image ', '')
    # If class_id is already a string starting with 'n', use it directly
    elif isinstance(class_id, str) and class_id.startswith('n'):
        formatted_id = class_id
    # If it's a number or string number, format it
    else:
        # Convert to integer if it's a string number
        if isinstance(class_id, str):
            class_id = int(class_id.strip('n'))
        formatted_id = f"n{class_id:08d}"
    
    # Read the word lemmas file
    with open('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/dataloader/imagenet21k_wordnet_lemmas.txt', 'r') as f:
        lemmas = [line.strip() for line in f]
    
    # Read the wordnet IDs file
    with open('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/dataloader/imagenet21k_wordnet_ids.txt', 'r') as f:
        ids = [line.strip() for line in f]
    
    # Create a dictionary mapping IDs to lemmas
    id_to_lemma = dict(zip(ids, lemmas))
    
    # Return the corresponding lemma/class name
    class_name = id_to_lemma.get(formatted_id, "Class ID not found")
    if ',' in class_name:
        class_name = class_name.split(',')[0]

    return class_name

def plot_images(images, image_indices, class_names, layer_idx, neuron_idx, model, save_dir, 
               type_of_sampling='max', activations=None, extreme_type='top', layer_type=None, 
               file_path=None, save_figures=SAVE_FIGURES, neuron_indices_mlp_out=None, neuron_indices_resid_post=None):
    
    
    grid_size = int(np.ceil(np.sqrt(len(images))))

    # reverse order
    images = images[::-1]
    image_indices = image_indices[::-1]
    class_names = class_names[::-1]
    

    if layer_type == 'hook_mlp_out':
        orig_neuron_index = neuron_indices_mlp_out[layer_idx][neuron_idx]
    elif layer_type == 'hook_resid_post':
        orig_neuron_index = neuron_indices_resid_post[layer_idx][neuron_idx]
    
    def create_figure(include_heatmap=True):

        fig, axs = plt.subplots(int(np.ceil(len(images)/grid_size)), grid_size, figsize=(20, 20))
        fig.suptitle(f"Neuron {orig_neuron_index}, Layer {layer_idx}, {extreme_type.capitalize()} Activations ({layer_type}, {type_of_sampling})", fontsize=16)
        
        axs = axs.flatten()
        for ax in axs:
            ax.axis('off')
        

        display_transform = get_clip_val_transforms_display()
        transform = get_clip_val_transforms()

        for idx, (image, image_idx, class_name) in enumerate(zip(images, image_indices, class_names)):
            if isinstance(image, Image.Image):
                display_image = display_transform(image)
            else:
                img = image.permute(1, 2, 0).numpy()
                display_image = (img - img.min()) / (img.max() - img.min())
                
            ax = axs[idx]

            if include_heatmap:
                layer_key = f'blocks.{layer_idx}.{layer_type}'
                ax.imshow(display_image)

   
                image = transform(image).to('cuda').unsqueeze(0)
                _, cache = model.run_with_cache(image, names_filter=[layer_key])
                all_activations = cache[layer_key][:,:,orig_neuron_index]
                pixel_num = int(np.sqrt(all_activations.shape[-1]-1))
                heatmap = image_patch_heatmap(all_activations.squeeze(0), pixel_num=pixel_num)
                ax.imshow(heatmap, cmap='viridis', alpha=0.3)
            else:
                ax.imshow(display_image)
            
            ax.axis('off')

            class_name = get_imagenet21k_class_name(class_name)
            
            fontsize = 14
            if activations is not None:
                ax.set_title(f"{class_name}  {activations[idx]:.4f}", fontsize=fontsize)
            else:
                ax.set_title(f"{class_name}", fontsize=fontsize)

        return fig, axs

    # Create and save figures
    fig_with_heatmap, _ = create_figure(include_heatmap=True)
    fig_without_heatmap, _ = create_figure(include_heatmap=False)
    
    base_name = f"neuron_{orig_neuron_index}_layer_{layer_idx}_{extreme_type}_{type_of_sampling}_{layer_type}"
    
    if save_figures:
        for fig, suffix in [(fig_with_heatmap, ""), (fig_without_heatmap, "_no_heatmap")]:
            for is_svg in [True, False]:
                if is_svg:
                    base_dir = f"{save_dir}/svg/layer_{layer_idx}/neuron_{neuron_idx}/{layer_type}/{type_of_sampling}/{extreme_type}"
                    file_ext = "svg"
                else:
                    base_dir = f"{save_dir}/layer_{layer_idx}/neuron_{neuron_idx}/{layer_type}/{type_of_sampling}/{extreme_type}"
                    file_ext = "png"
                
                os.makedirs(base_dir, exist_ok=True)
                save_path = f"{base_dir}/{base_name}{suffix}.{file_ext}"
                
                fig.tight_layout()
                fig.savefig(save_path, format=file_ext, dpi=300 if file_ext == 'png' else None, bbox_inches='tight')
                if VERBOSE:
                    print(f"Saved {file_ext.upper()} image to {save_path}")

    plt.close('all')


def image_patch_heatmap(activation_values, image_size=224, pixel_num=7):
    activation_values = activation_values.detach().cpu().numpy()
    activation_values = activation_values[1:]
    activation_values = activation_values.reshape(pixel_num, pixel_num)
    heatmap = np.zeros((image_size, image_size))
    patch_size = image_size // pixel_num
    for i in range(pixel_num):
        for j in range(pixel_num):
            heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = activation_values[i, j]
    return heatmap
def visualize_layer_activations(checkpoint_path, save_dir, image_path, model, layer_idx):
    checkpoint = torch.load(checkpoint_path)
    
    # load neuron indices
    neuron_indices_mlp_out = np.load('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/clip_base_mlp_out.npy', allow_pickle=True).item()
    neuron_indices_resid_post = np.load('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/clip_base_residual_post.npy', allow_pickle=True).item()
    
    # Restructure the processing to handle both layer types for each neuron
    for strategy in checkpoint.keys():
        for interval in checkpoint[strategy].keys():
            # Get data for both layer types
            layer_data = {
                layer_type: checkpoint[strategy][interval][layer_type]
                for layer_type in checkpoint[strategy][interval].keys()
            }
            
            # Determine the number of neurons to process
            num_neurons = min(15, min(
                layer_data[layer_type]['activations'][layer_idx].shape[0]
                for layer_type in layer_data.keys()
            ))
            
            print(f"Processing layer {layer_idx} for {strategy}/{interval}")
            for neuron_idx in tqdm(range(num_neurons)):
                # Process each layer type for this neuron
                for layer_type, data in layer_data.items():
                    activations = data['activations'][layer_idx][neuron_idx]
                    image_ids = data['image_ids'][layer_idx][neuron_idx]
                    
                    # Load images
                    images = []
                    class_names = []
                    for img_id in image_ids:
                        img_path = f"{image_path}/{img_id}.jpg"
                        if os.path.exists(img_path):
                            img = Image.open(img_path)
                            images.append(img)
                            class_names.append(f"Image {img_id}")
                    
                    if images:  # Only plot if we have images
                        plot_images(
                            images=images,
                            image_indices=image_ids,
                            class_names=class_names,
                            layer_idx=layer_idx,
                            neuron_idx=neuron_idx,
                            model=model,
                            save_dir=save_dir,
                            type_of_sampling=strategy,
                            activations=activations,
                            extreme_type=interval,
                            layer_type=layer_type,
                            save_figures=SAVE_FIGURES,
                            neuron_indices_mlp_out=neuron_indices_mlp_out,
                            neuron_indices_resid_post=neuron_indices_resid_post,
                        )

# [Rest of the functions remain the same]

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize neuron activations for a specific layer')
    parser.add_argument('--layer', type=int, required=True, help='Layer number to process')
    parser.add_argument('--checkpoint', type=str, default='/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/activation_results/checkpoint_final.pt')
    parser.add_argument('--image-dir', type=str, default='/network/scratch/s/sonia.joseph/CLIP_AUDIT/selected_imagenet21k')
    parser.add_argument('--save-dir', type=str, 
                        default="/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons")
    
    args = parser.parse_args()
    
    model_name = 'open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K'
    model = HookedViT.from_pretrained(model_name, is_clip=True, is_timm=False, fold_ln=False).to('cuda')
    
    print(f"Processing layer {args.layer}")
    visualize_layer_activations(
        args.checkpoint,
        args.save_dir,
        args.image_dir,
        model=model,
        layer_idx=args.layer
    )

if __name__ == "__main__":
    main()