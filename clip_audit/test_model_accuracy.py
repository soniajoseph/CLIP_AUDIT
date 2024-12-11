from datasets import load_dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
import torch

# Set up paths
root_dir = Path("/network/datasets/conceptualcaptions")  # CHANGE THIS to your actual path
tsv_path = root_dir / "Image_Labels_Subset_Train_GCC-Labels-training.tsv"
train_dir = root_dir / "Train"

# Load the TSV
dataset = load_dataset(
    'csv',
    data_files={'train': str(tsv_path)},
    delimiter='\t',
    column_names=['image_id', 'caption']
)

# Function to load image from local path
def load_image(example):
    image_id = str(example['image_id'])
    img_path = train_dir / image_id[:3] / f"{image_id}-0.jpg"
    try:
        image = Image.open(img_path).convert('RGB')
        return {'image': image}
    except Exception as e:
        return {'image': None}

# Add images to the dataset
dataset = dataset['train'].map(
    load_image,
    num_proc=4,
    remove_columns=dataset['train'].column_names  # Prevent duplicate columns
)

# Remove examples where image loading failed
dataset = dataset.filter(lambda x: x['image'] is not None)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Add transforms to the dataset
def apply_transforms(example):
    return {
        'image': transform(example['image']),
        'caption': example['caption']
    }

dataset.set_transform(apply_transforms)

# Create train/val split
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Create DataLoaders if needed
train_loader = DataLoader(
    dataset['train'],
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    dataset['test'],
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True if torch.cuda.is_available() else False
)

# Example usage:
# Print first batch
for batch in train_loader:
    print("Image shape:", batch['image'].shape)  # Should be [batch_size, 3, 224, 224]
    print("Sample caption:", batch['caption'][0])
    break