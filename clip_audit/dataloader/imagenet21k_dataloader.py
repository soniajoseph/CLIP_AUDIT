# import torch
# from torch.utils.data import IterableDataset, DataLoader
# from PIL import Image
# import tarfile
# import io
# from typing import Optional, Callable
# import mmap
# import os
# import pickle

# import torch
# from torch.utils.data import IterableDataset, DataLoader
# from PIL import Image
# import tarfile
# import io
# from typing import Optional, Callable
# import mmap
# import os
# import pickle
# import time
# from tqdm import tqdm


# import torch
# from torch.utils.data import IterableDataset, DataLoader, Dataset
# from PIL import Image
# import tarfile
# import io
# from typing import Optional, Callable
# import pickle
# import os
# from tqdm import tqdm

# import torch
# from torch.utils.data import IterableDataset, DataLoader, Dataset
# from PIL import Image
# import tarfile
# import io
# from typing import Optional, Callable
# import pickle
# import os
# from tqdm import tqdm

# class TarImageNet21kIndexer:
#     def __init__(self, tar_path: str, index_path: Optional[str] = None):
#         self.tar_path = tar_path
#         self.index_path = index_path
#         self.offsets = []
#         self.filenames = []

#     def build_index(self):
#         print("Building index... this will take a while but only needs to be done once")
#         with tarfile.open(self.tar_path, 'r:gz') as tar:
#             offset = 0
#             for member in tqdm(tar):
#                 if member.name.endswith(('.JPEG', '.jpg', '.jpeg', '.PNG', '.png')):
#                     self.offsets.append(offset)
#                     self.filenames.append(member.name)
#                 offset = tar.offset
        
#         # Save index
#         with open(self.index_path, 'wb') as f:
#             pickle.dump((self.offsets, self.filenames), f)
#         print(f"Index saved to {self.index_path}")
#         return self.offsets, self.filenames

#     def get_index(self):
#         if os.path.exists(self.index_path):
#             print(f"Loading existing index from {self.index_path}")
#             with open(self.index_path, 'rb') as f:
#                 self.offsets, self.filenames = pickle.load(f)
#         else:
#             self.offsets, self.filenames = self.build_index()
#         return self.offsets, self.filenames

# class IndexedTarImageNet21k(Dataset):
#     def __init__(self, tar_path: str, transform: Optional[Callable] = None, index_path: Optional[str] = None):
#         self.tar_path = tar_path
#         self.transform = transform
        
#         # Load or build index if necessary
#         indexer = TarImageNet21kIndexer(tar_path, index_path)
#         self.offsets, self.filenames = indexer.get_index()
        
#         # Keep tar file handle open for faster access
#         self.tar = tarfile.open(tar_path, 'r:gz')

#     def __len__(self):
#         return len(self.offsets)

#     def __getitem__(self, idx):
#         # Seek to correct position
#         self.tar.offset = self.offsets[idx]
        
#         # Read image
#         member = self.tar.next()
#         f = self.tar.extractfile(member)
#         if f is None:
#             raise ValueError(f"Could not read file at index {idx}")
            
#         img = Image.open(io.BytesIO(f.read())).convert('RGB')
#         if self.transform:
#             img = self.transform(img)
#         return img

#     def __del__(self):
#         if hasattr(self, 'tar'):
#             self.tar.close()
            
# # Usage example:
# from torchvision import transforms
# import matplotlib.pyplot as plt

# def show_samples(dataset_path, num_samples=10):

#     transform_vis = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#     ])
    
#     # Test with just 100 images:
#     dataset = IndexedTarImageNet21k(
#         tar_path=dataset_path,
#         transform=transform_vis,
#         index_path = '/network/scratch/s/sonia.joseph/datasets/imagenet21k/index.idx'
#     )
        
#     dataloader = DataLoader(
#         dataset, 
#         batch_size=1, 
#         num_workers=0
#     )
    
#     fig, axes = plt.subplots(2, 5, figsize=(15, 6))
#     axes = axes.ravel()
    
#     for idx, (img, label, index) in enumerate(dataloader):
#         if idx >= num_samples:
#             break
        
#         img = img.squeeze(0).permute(1, 2, 0).numpy()
#         axes[idx].imshow(img)
#         axes[idx].axis('off')
        
#         idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
#         class_name = idx_to_class[label.item()]
#         axes[idx].set_title(f'Class: {class_name}', fontsize=8)
    
#     plt.tight_layout()
#     plt.show()

# # # Example usage:
# # if __name__ == "__main__":
# #     dataset_path = '/network/datasets/imagenet21k/winter21_whole.tar.gz'
# #     show_samples(dataset_path)