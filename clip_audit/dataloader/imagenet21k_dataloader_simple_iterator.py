import tarfile
import io
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import json
import time
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import tarfile
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

import tarfile
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import time

import tarfile
import io
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import time

class StreamingImageNet21k(IterableDataset):
    def __init__(self, tar_path, transform=None, seed=42):
        self.tar_path = tar_path
        self.transform = transform
        self.seed = seed

    
    def _get_stream(self):
        worker_info = torch.utils.data.get_worker_info()
        tar = tarfile.open(self.tar_path, 'r|gz')  # Streaming mode
        
        current_class_tar = None
        images_yielded = 0
        
        try:
            for member in tar:
                if member.name.endswith('.tar'):
                    # Close previous class tar if exists
                    if current_class_tar is not None:
                        current_class_tar.close()
                    
                    # Extract and open new class tar
                    class_id = member.name.split('/')[-1].split('.')[0]
                    inner_tar_data = tar.extractfile(member)
                    current_class_tar = tarfile.open(fileobj=io.BytesIO(inner_tar_data.read()))
                    
                    # Get all images from this class tar
                    for img_member in current_class_tar.getmembers():
                        if not img_member.name.endswith(('.JPEG', '.jpg', '.jpeg', '.png')):
                            continue
                            
                        # Worker sharding - only process images for this worker
                        if worker_info is not None:
                            if images_yielded % worker_info.num_workers != worker_info.id:
                                images_yielded += 1
                                continue
                        
                        try:
                            # Load and process image
                            img_data = current_class_tar.extractfile(img_member)
                            img = Image.open(io.BytesIO(img_data.read()))
                            
                            if self.transform:
                                img = self.transform(img)
                            
                            yield {
                                'image': img,
                                'class_id': class_id,
                                'image_id': img_member.name.split('.')[0],
                                'index': images_yielded
                            }
                            
                        except Exception as e:
                            print(f"Error processing image {img_member.name}: {e}")
                            
                        images_yielded += 1
            
        finally:
            if current_class_tar is not None:
                current_class_tar.close()
            tar.close()

    def find_image(self, target_total_id):
        """Find and load a specific image by image_id"""
        tar = tarfile.open(self.tar_path, 'r|gz')

        target_class_id = target_total_id.split('_')[0]
        target_image_id = target_total_id.split('_')[1]
        print(f"Looking for image {target_image_id} in class {target_class_id}")
        
        try:
            current_class_tar = None
            for member in tar:
                if member.name.endswith('.tar'):
                    # Close previous class tar if exists
                    if current_class_tar is not None:
                        current_class_tar.close()
                    
                    # Extract and open new class tar
                    class_id = member.name.split('/')[-1].split('.')[0]

                    if class_id != target_class_id:
                        # print(f"Skipping class {class_id}, looking for {target_class_id}")
                        continue
                    
                    # print(f"Found matching class tar: {member.name}")
                
                    inner_tar_data = tar.extractfile(member)
                    current_class_tar = tarfile.open(fileobj=io.BytesIO(inner_tar_data.read()))
                    
                    # Look for the target image in this class
                    for img_member in current_class_tar.getmembers():
                        # print(f"Checking image {img_member.name}")
                        if img_member.name.endswith(('.JPEG', '.jpg', '.jpeg', '.png')):
                            current_image_id = img_member.name.split('_')[-1]
                            current_image_id = current_image_id.split('.')[0]

                            # print(f"On image {current_image_id}")
                            
                            if current_image_id == target_image_id:
                                # Found the target image
                                img_data = current_class_tar.extractfile(img_member)
                                img = Image.open(io.BytesIO(img_data.read())).convert('RGB')
                                
                                if self.transform:
                                    img = self.transform(img)
                                
                                print(f"Found image {target_image_id} in class {class_id}")
                                
                                return {
                                    'image': img,
                                    'class_id': class_id,
                                    'image_id': current_image_id
                                }
        
        finally:
            if current_class_tar is not None:
                current_class_tar.close()
            tar.close()
        
        raise ValueError(f"Image ID {target_image_id} not found")
    
    def __iter__(self):
        return self._get_stream()


def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    class_ids = [item['class_id'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    indices = [item['index'] for item in batch]
    
    return {
        'image': images,
        'class_id': class_ids,
        'image_id': image_ids,
        'index': indices
    }


def load_imagenet21k(tar_path, transforms, batch_size=64, num_workers=4, pin_memory=True):
    dataset  = StreamingImageNet21k(tar_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader

# import h5py
# import h5repack

# # Path to your corrupted file
# input_file = '/network/scratch/s/sonia.joseph/CLIP_AUDIT/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/conceptual_captions/train/hook_mlp_out.h5'
# # Path for the repaired file
# output_file = input_file.replace('.h5', '_repaired.h5')

# h5repack.repack(input_file, output_file)
# print("File successfully repaired")



# # Usage example:
# if __name__ == "__main__":
#     tar_path = '/network/datasets/imagenet21k/winter21_whole.tar.gz'
#     cache_dir = '/network/scratch/s/sonia.joseph/datasets/imagenet21k'
    
#     # Create dataset
#     # dataset = ImageNet21kDataset(tar_path, cache_dir=cache_dir)

#     from clip_audit.utils.transforms import get_clip_transforms

#     transform =get_clip_transforms()
#     dataset = StreamingImageNet21k(tar_path, transform=transform)

#     print("Finding specific image")
#     target_id = 'n02689434_9032'

#     from time import time
#     start = time()
#     target_image = dataset.find_image(target_id)
#     print(f"Found image in {time() - start:.2f} seconds")
#     # display
#     plt.figure()
#     plt.title(f'{target_id}')
#     plt.imshow(target_image['image'].numpy().transpose(1, 2, 0))
#     plt.savefig('imagenet21k_target.png')

    
    # # Create dataloader
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=collate_fn
    # )
    
    # # # Example iteration
    # # for i, batch in enumerate(dataloader):
    # #     if i >= 2:  # Just show first 2 batches
    # #         break
            
    # #     print(f"Batch {i}:")
    # #     print(f"Images shape: {batch['image'].shape}")
    # #     print(f"Class IDs: {batch['class_id']}")
    # #     print(f"Image IDs: {batch['image_id']}")
    # #     print(f"Indices: {batch['index']}")
    # #     print("---")


    
    # # Visualization example
    # def show_batch(batch, num_images=4):
    #     images = batch['image'][:num_images]
    #     class_ids = batch['class_id'][:num_images]
    #     indices = batch['index'][:num_images]
        
    #     plt.figure(figsize=(15, 4))
    #     for i, (img, class_id, idx) in enumerate(zip(images, class_ids, indices)):
    #         plt.subplot(1, num_images, i + 1)
    #         img = img.numpy().transpose(1, 2, 0)
    #         # Denormalize
    #         img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    #         img = np.clip(img, 0, 1)
            
    #         plt.imshow(img)
    #         plt.title(f"Class: {class_id}\nIdx: {idx}")
    #         plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig('imagenet21k_batch.png')
    #     plt.show()
    
    # # Show first batch
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=4,
    #     shuffle=False,
    #     num_workers=0,  # Use 0 for visualization
    #     collate_fn=collate_fn
    # )
    
    # batch = next(iter(dataloader))
    # show_batch(batch)

    
