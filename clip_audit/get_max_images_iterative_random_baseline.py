# Load model
from vit_prisma.models.base_vit import HookedViT
import numpy as np
from clip_audit.dataloader.imagenet21k_dataloader_simple_iterator import load_imagenet21k
from vit_prisma.models.base_vit import HookedViT
from clip_audit.utils.transforms import get_clip_transforms
from tqdm import tqdm
import torch
import os

# Constants
K = 20  # number of top activations to keep
BATCH_SIZE = 64
DEVICE = 'cuda'
MAX = 14_300_000  # Max number of images to process
SAVE_EVERY = 10000  # Save every 1000 batches
RANDOM = True


# Create save directory if it doesn't exist
save_dir = 'activation_results'
os.makedirs(save_dir, exist_ok=True)

# Load Data
transforms = get_clip_transforms()
tar_path = '/network/datasets/imagenet21k/winter21_whole.tar.gz'
dataloader = load_imagenet21k(tar_path, transforms=transforms, batch_size=BATCH_SIZE)

# Load indices 
neuron_indices_mlp_out = np.load('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/clip_base_mlp_out.npy', allow_pickle=True).item()
neuron_indices_resid_post = np.load('/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/saved_data/clip_base_residual_post.npy', allow_pickle=True).item()

# Load model
model_name = 'open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K'
model = HookedViT.from_pretrained(model_name, is_timm=False, is_clip=True, fold_ln=False, center_writing_weights=False)
model.to(DEVICE)
print(f"Model loaded: {model_name}")

if RANDOM:
    print("Randomizing model weights for random baseline")
    cfg = model.cfg
    random_model = HookedViT(cfg)
    random_model.init_weights()

    del model
    model = random_model
    model.to(DEVICE)

num_layers = 1
num_neurons = len(neuron_indices_mlp_out[0])
sublayer_types = ['hook_resid_post', 'hook_mlp_out']
intervals = ['top', 'bottom']
sampling_strategies = ['max', 'avg', 'max_cls']

print(f"Number of layers: {num_layers}")
print(f"Number of neurons: {num_neurons}")

def convert_to_uint8(images):
    """Convert float tensor images to uint8 format"""
    return (images.cpu() * 255).to(torch.uint8).clone()

def convert_to_float(images):
    """Convert uint8 images back to float format"""
    return images.float() / 255.0

# Initialize storage using torch tensors (using float32 for activations as they need precision)
top_k_storage = {
    strategy: {
        interval: {
            sublayer: {
                'activations': torch.full((num_layers, num_neurons, K), float('-inf'), device=DEVICE),
                'indices': torch.zeros((num_layers, num_neurons, K), dtype=torch.long, device=DEVICE),
                'image_ids': [[[None for _ in range(K)] for _ in range(num_neurons)] for _ in range(num_layers)],
                # 'images': [[[None for _ in range(K)] for _ in range(num_neurons)] for _ in range(num_layers)]
            } for sublayer in sublayer_types
        } for interval in intervals
    } for strategy in sampling_strategies
}

# Create hooks for collecting activations
hooks_to_run = []
for layer_idx in range(num_layers):
    hooks_to_run.extend([
        f'blocks.{layer_idx}.hook_resid_post',
        f'blocks.{layer_idx}.hook_mlp_out'
    ])

def save_checkpoint(storage, sample_num):
    checkpoint_path = os.path.join(save_dir, f'checkpoint_{sample_num}.pt')
    
    cpu_storage = {
        strategy: {
            interval: {
                sublayer: {
                    'activations': layer_data['activations'].cpu(),
                    'indices': layer_data['indices'].cpu(),
                    'image_ids': layer_data['image_ids'],
                    # 'images': layer_data['images']  # Already in uint8 format on CPU
                } for sublayer, layer_data in interval_data.items()
            } for interval, interval_data in strategy_data.items()
        } for strategy, strategy_data in storage.items()
    }
    
    torch.save(cpu_storage, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

count = 0
with torch.no_grad():
    for batch_idx, output in tqdm(enumerate(dataloader), total = MAX // BATCH_SIZE, desc="Processing batches"):
        if output['image'].shape[0] != BATCH_SIZE:  # Skip incomplete batches
            continue
            
        images = output['image'].to(DEVICE)
        image_ids = output['image_id']
        
        # Run model with hooks
        _, cache = model.run_with_cache(images, names_filter=hooks_to_run)
        
        # Process each layer
        for layer_idx in range(num_layers):
            for sublayer_type in sublayer_types:
                hook_name = f'blocks.{layer_idx}.{sublayer_type}'
                activations = cache[hook_name]
                input_size = activations.shape[0]
                
                neuron_indices = (neuron_indices_mlp_out if 'mlp_out' in sublayer_type 
                                else neuron_indices_resid_post)
                neuron_indices = neuron_indices[layer_idx]
                
                # Calculate activations for each strategy
                activation_dict = { 
                    'max': activations.max(1)[0],
                    'avg': activations.mean(1),
                    'max_cls': activations[:, 0]
                }
                
                for strategy in activation_dict.keys():
                    selected_activations = activation_dict[strategy][:, neuron_indices].T
                    
                    for interval in intervals:
                        act_for_selection = selected_activations if interval == 'top' else -selected_activations
                        
                        # Get current storage
                        current_activations = top_k_storage[strategy][interval][sublayer_type]['activations'][layer_idx]
                        current_indices = top_k_storage[strategy][interval][sublayer_type]['indices'][layer_idx]
                        current_image_ids = top_k_storage[strategy][interval][sublayer_type]['image_ids'][layer_idx]
                        # current_images = top_k_storage[strategy][interval][sublayer_type]['images'][layer_idx]
                        
                        # Combine current and new activations
                        combined_activations = torch.cat([
                            current_activations,
                            act_for_selection
                        ], dim=1)
                        
                        combined_indices = torch.cat([
                            current_indices,
                            torch.arange(batch_idx * input_size, (batch_idx + 1) * input_size, 
                                       device=DEVICE).repeat(num_neurons, 1)
                        ], dim=1)
                        
                        # Get top K
                        topk_values, topk_indices = torch.topk(
                            combined_activations,
                            k=K,
                            dim=1,
                            largest=True
                        )
                        
                        if interval == 'bottom':
                            topk_values = -topk_values
                        
                        # Update storage
                        top_k_storage[strategy][interval][sublayer_type]['activations'][layer_idx] = topk_values
                        top_k_storage[strategy][interval][sublayer_type]['indices'][layer_idx] = torch.gather(
                            combined_indices, 1, topk_indices)
                        
                        # Update image_ids and images
                        for neuron_idx in range(num_neurons):
                            selected_indices = topk_indices[neuron_idx].cpu().numpy()
                            new_image_ids = []
                            # new_images = []
                            
                            for idx in selected_indices:
                                if idx < len(current_image_ids[neuron_idx]):
                                    new_image_ids.append(current_image_ids[neuron_idx][idx])
                                    # new_images.append(current_images[neuron_idx][idx])
                                else:
                                    batch_idx_local = idx - len(current_image_ids[neuron_idx])
                                    new_image_ids.append(image_ids[batch_idx_local])
                                    # Store as uint8 to save memory while preserving quality
                                    # new_images.append(convert_to_uint8(images[batch_idx_local]))
                            
                            top_k_storage[strategy][interval][sublayer_type]['image_ids'][layer_idx][neuron_idx] = new_image_ids
                            # top_k_storage[strategy][interval][sublayer_type]['images'][layer_idx][neuron_idx] = new_images

        count += input_size
        if count > MAX:
            break

        # Periodic saving and memory clearing
        if batch_idx % SAVE_EVERY == 0 and batch_idx > 0:
            save_checkpoint(top_k_storage, count)
            torch.cuda.empty_cache()

# Save final results
final_path = os.path.join(save_dir, f'random_top_k_activations_final_{MAX}.pt')
save_checkpoint(top_k_storage, 'final')
print(f"Final results saved to {final_path}")
