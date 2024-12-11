import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd

import matplotlib.pyplot as plt


# import dataloader

class ConceptualCaptionsDataset(Dataset):
    def __init__(self, tsv_file, image_root_dir, transform=None):  # Added max_tags parameter
        """
        Args:
            tsv_file (string): Path to the TSV file with annotations
            image_root_dir (string): Directory with all the image folders
            transform (callable, optional): Optional transform to be applied on an image
            max_tags (int): Maximum number of tags to include (will pad if less)
        """
        self.annotations = pd.read_csv(tsv_file, sep='\t', header=None)
        print(f"Found {len(self.annotations)} annotations in TSV file")

        self.image_root_dir = image_root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        print("Scanning image directories...")
        self.image_files = []
        folders = sorted([f for f in os.listdir(image_root_dir) if os.path.isdir(os.path.join(image_root_dir, f))])
        print(f"Found {len(folders)} folders")
        
        for folder in folders:
            folder_path = os.path.join(image_root_dir, folder)
            image_files = sorted([f for f in os.listdir(folder_path)])
            self.image_files.extend([os.path.join(folder, img) for img in image_files])
            
        self.image_files.sort(key=lambda x: int(x.split('-')[0].split('/')[-1]))
        print(f"Found {len(self.image_files)} total images")
        
        if len(self.image_files) == 0:
            raise ValueError("No images found in the specified directory")

        self.valid_indices = []
        img_counter = 0
        for i in range(len(self.annotations)):
            if img_counter < len(self.image_files):
                img_id = int(self.image_files[img_counter].split('-')[0].split('/')[-1])
                if i == img_id:
                    self.valid_indices.append(i)
                    img_counter += 1
                    
        print(f"Found {len(self.valid_indices)} valid image-annotation pairs")

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

        sample = {
            'image': image,
            'caption': caption,
            'url': url
        }

        return sample
        


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
    image_root_dir=parent_dir
)


dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

# Get first batch
batch = next(iter(dataloader))

# Set up the plot
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

# Convert tensor images and display
for i in range(10):
    img = batch['image'][i]
    # Convert tensor to numpy array and transpose to correct format
    img = img.permute(1, 2, 0).numpy()
    # Normalize to [0,1] range if needed
    img = (img - img.min()) / (img.max() - img.min())
    
    axes[i].imshow(img)
    axes[i].axis('off')
    caption = batch['caption'][i][:50] + '...' if len(batch['caption'][i]) > 50 else batch['caption'][i]
    axes[i].set_title(caption, fontsize=8)

plt.tight_layout()
plt.savefig('first_10_images_from_dataloader.png')
plt.close()

print("Images have been saved as 'first_10_images_from_dataloader.png'")