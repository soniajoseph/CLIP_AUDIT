import torch
import os
from tqdm import tqdm
from clip_audit.dataloader.imagenet21k_dataloader_simple_iterator import load_imagenet21k
from clip_audit.utils.transforms import get_clip_transforms
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def denormalize_clip(tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(tensor.device)
    return torch.clamp((tensor * std) + mean, 0, 1)

def save_image(args):
    img_tensor, image_id, save_folder, to_pil = args
    img = to_pil(img_tensor.cpu())
    img.save(os.path.join(save_folder, f"{image_id}.jpg"))

def save_matching_images(image_ids_to_find, tar_path, save_folder, batch_size=256, num_workers=8):
    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(save_folder, exist_ok=True)
    transform = get_clip_transforms()
    ids_to_find = set(image_ids_to_find)
    print(f"Starting search for {len(ids_to_find)} images")
    
    # Increase batch size and num_workers for faster loading
    dataloader = load_imagenet21k(
        tar_path, 
        transforms=transform, 
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    found_count = 0
    to_pil = transforms.ToPILImage()
    
    # Create a thread pool for parallel image saving
    max_threads = min(32, multiprocessing.cpu_count() * 2)
    pool = ThreadPoolExecutor(max_workers=max_threads)
    
    # Create a batch of save tasks
    save_tasks = []
    
    for output in tqdm(dataloader, desc="Searching images", total=14_200_000 // batch_size):
        images = output['image'].to(device)
        batch_image_ids = output['image_id']
        
        # Process batch
        for idx, image_id in enumerate(batch_image_ids):
            if image_id in ids_to_find:
                img_tensor = denormalize_clip(images[idx])
                
                # Add save task to queue
                save_tasks.append((img_tensor, image_id, save_folder, to_pil))
                
                ids_to_find.remove(image_id)
                found_count += 1
                
                # Process save tasks in batches
                if len(save_tasks) >= 100:
                    list(pool.map(save_image, save_tasks))
                    save_tasks = []
                
                if found_count % 100 == 0:
                    print(f"Found {found_count} images. {len(ids_to_find)} remaining.")
        
        if len(ids_to_find) == 0:
            print("All images found!")
            break
    
    # Process remaining save tasks
    if save_tasks:
        list(pool.map(save_image, save_tasks))
    
    pool.shutdown()
    
    print(f"Search complete. Found {found_count} images.")
    if len(ids_to_find) > 0:
        print(f"Could not find {len(ids_to_find)} images.")
        with open(os.path.join(save_folder, 'unfound_ids.txt'), 'w') as f:
            for id_ in ids_to_find:
                f.write(f"{id_}\n")



# Usage
tar_path = '/network/datasets/imagenet21k/winter21_whole.tar.gz'
checkpoint_path = '/home/mila/s/sonia.joseph/CLIP_AUDIT/clip_audit/activation_results/image_ids.pt'
save_folder = '/network/scratch/s/sonia.joseph/CLIP_AUDIT/selected_imagenet21k'

# First load the image IDs from checkpoint
image_ids = torch.load(checkpoint_path)

print(f"Loaded {len(image_ids)} image IDs")
print(f"First ten {image_ids[:10]}")
# Then search and save the images
# Use optimized batch size and num_workers
save_matching_images(
    image_ids, 
    tar_path, 
    save_folder, 
    batch_size=4096,  # Increased batch size
    num_workers=8    # Adjust based on your CPU cores
)