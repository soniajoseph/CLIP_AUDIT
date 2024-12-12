import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd

import matplotlib.pyplot as plt

from torchvision.transforms import InterpolationMode


import hashlib

import pickle

import os
import torch
import pandas as pd
import hashlib
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

class ConceptualCaptionsDataset(Dataset):
    def __init__(self, train_or_val, tsv_file, image_root_dir, transform=None, cache_dir='/network/scratch/s/sonia.joseph/datasets/conceptual_captions'):
        """
        Args:
            tsv_file (string): Path to the TSV file with annotations
            image_root_dir (string): Directory with all the image folders
            transform (callable, optional): Optional transform to be applied on an image
            cache_dir (string, optional): Directory to store cached files (e.g. valid image-annotation pairs)
        """
        self.annotations = pd.read_csv(tsv_file, sep='\t', header=None)
        print(f"Found {len(self.annotations)} annotations in TSV file")

        self.image_root_dir = image_root_dir
        self.cache_dir = os.path.join(cache_dir, train_or_val)
        os.makedirs(self.cache_dir, exist_ok=True)

        if transform is None:
            normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        # Generate cache file paths
        cache_hash = hashlib.md5(f"{tsv_file}_{image_root_dir}".encode()).hexdigest()
        self.image_files_cache = os.path.join(self.cache_dir, f'image_files_{cache_hash}.pkl')
        self.valid_indices_cache = os.path.join(self.cache_dir, f'valid_indices_{cache_hash}.pkl')

        # Load or create image files list
        self.image_files = self._load_or_create_image_files()
        
        # Load or create valid indices
        self.valid_indices = self._load_or_create_valid_indices()

    def _safe_load_cache(self, cache_file):
        """Safely load cache file with error handling."""
        try:
            if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except (EOFError, pickle.UnpicklingError, Exception) as e:
            print(f"Cache file corrupted or invalid: {e}")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"Deleted corrupted cache file: {cache_file}")
        return None

    def _load_or_create_image_files(self):
        """Load image files from cache or create if not exists."""
        cached_data = self._safe_load_cache(self.image_files_cache)
        if cached_data is not None:
            print("Successfully loaded image files from cache")
            return cached_data

        print("Scanning image directories...")
        image_files = []
        folders = sorted([f for f in os.listdir(self.image_root_dir) 
                        if os.path.isdir(os.path.join(self.image_root_dir, f))])
        print(f"Found {len(folders)} folders")
        
        for folder in folders:
            folder_path = os.path.join(self.image_root_dir, folder)
            folder_images = sorted([f for f in os.listdir(folder_path)])
            image_files.extend([os.path.join(folder, img) for img in folder_images])
            
        image_files.sort(key=lambda x: int(x.split('-')[0].split('/')[-1]))
        print(f"Found {len(image_files)} total images")
        
        if len(image_files) == 0:
            raise ValueError("No images found in the specified directory")

        # Cache the results
        try:
            with open(self.image_files_cache, 'wb') as f:
                pickle.dump(image_files, f)
            print("Successfully cached image files")
        except Exception as e:
            print(f"Failed to cache image files: {e}")

        return image_files

    def _load_or_create_valid_indices(self):
        """Load valid indices from cache or create if not exists."""
        cached_data = self._safe_load_cache(self.valid_indices_cache)
        if cached_data is not None:
            print(f"Successfully loaded {len(cached_data)} valid image-annotation pairs from cache")
            return cached_data

        print("Creating valid indices...")
        valid_indices = []
        img_counter = 0
        for i in range(len(self.annotations)):
            if img_counter < len(self.image_files):
                img_id = int(self.image_files[img_counter].split('-')[0].split('/')[-1])
                if i == img_id:
                    valid_indices.append(i)
                    img_counter += 1

        print(f"Found {len(valid_indices)} valid image-annotation pairs")

        # Cache the results
        try:
            with open(self.valid_indices_cache, 'wb') as f:
                pickle.dump(valid_indices, f)
            print("Successfully cached valid indices")
        except Exception as e:
            print(f"Failed to cache valid indices: {e}")

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ann_idx = self.valid_indices[idx]
        img_path = os.path.join(self.image_root_dir, self.image_files[idx])
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        caption = self.annotations.iloc[ann_idx, 0]
        url = self.annotations.iloc[ann_idx, 1]

        return {
            'image': image,
            'caption': caption,
            'url': url
        }
        


def load_conceptual_captions(train_or_val,dataloader=True):
    if train_or_val == 'train':
        print("Loading training dataset")
        tsv_file = '/network/datasets/conceptualcaptions/Train/GCC-training.tsv'
        image_root_dir = '/network/datasets/conceptualcaptions/Train'
    elif train_or_val == 'val':
        print("Loading validation dataset")
        tsv_file = '/network/datasets/conceptualcaptions/Validation/GCC-1.1.0-Validation.tsv'
        image_root_dir = '/network/datasets/conceptualcaptions/Validation'
    cache_dir = '/network/scratch/s/sonia.joseph/datasets/conceptual_captions'
    dataset= ConceptualCaptionsDataset(train_or_val, tsv_file=tsv_file, image_root_dir=image_root_dir, cache_dir=cache_dir)
    if dataloader:
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        return dataloader
    else:
        return dataset


def test():
    # Example usage:

    parent_dir = '/network/datasets/conceptualcaptions/Train'
    tsv_file_name = 'GCC-training.tsv'


    # Set display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # Load TSV file
    tsv_path = os.path.join(parent_dir, tsv_file_name)
    df = pd.read_csv(tsv_path, sep='\t', header=None)

    # Print information
    print("\nFirst 5 rows of TSV file:")
    print(df.head(10))
    print("\nShape of TSV:", df.shape)
    print("\nColumn indices:", list(df.columns))


    # Create dataset
    dataset = ConceptualCaptionsDataset(
        tsv_file=os.path.join(parent_dir, tsv_file_name),
        image_root_dir=parent_dir,
        train_or_val='val'
    )


    clip_preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    # Get first batch
    batch = next(iter(dataloader))

    # Set up the plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()

    dataset = dataloader.dataset

    for i in range(10):

        img = dataset[i]['image']

        # img_pil = transforms.ToPILImage()(img)
        # img_pil = transforms.Resize(224, interpolation=InterpolationMode.BICUBIC)(img_pil)
        # img = transforms.CenterCrop(224)(img_pil)

        print(img.shape)

        img = img.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[i].imshow(img)
        axes[i].axis('off')
        caption = batch['caption'][i][:50] + '...' if len(batch['caption'][i]) > 50 else batch['caption'][i]
        axes[i].set_title(caption, fontsize=8)

    plt.tight_layout()
    plt.savefig('first_10_images_from_dataloader.png')
    plt.close()

    print("Images have been saved as 'first_10_images_from_dataloader.png'")