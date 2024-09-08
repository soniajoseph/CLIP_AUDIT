import torch
from torchvision import datasets, transforms
import os
import csv
from PIL import Image

from torchvision.datasets import ImageFolder
from typing import Any, Callable, Optional, Tuple

import torchvision

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from PIL import Image

class ImageNetValDataset(torch.utils.data.Dataset):
        def __init__(self, images_dir, imagenet_class_index, labels,  transform=None, return_index=True):
            self.images_dir = images_dir
            self.transform = transform
            self.labels = {}
            self.return_index = return_index


            # load label code to index
            self.label_to_index = {}
    
            with open(imagenet_class_index, 'r') as file:
                # Iterate over each line in the file
                for line_num, line in enumerate(file):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ')
                    code = parts[0]
                    self.label_to_index[code] = line_num


            # load image name to label code
            self.image_name_to_label = {}

            # Open the CSV file for reading
            with open(labels, mode='r') as csv_file:
                # Create a CSV reader object
                csv_reader = csv.DictReader(csv_file)
                
                # Iterate over each row in the CSV
                for row in csv_reader:
                    # Split the PredictionString by spaces and take the first element
                    first_prediction = row['PredictionString'].split()[0]
                    # Map the ImageId to the first part of the PredictionString
                    self.image_name_to_label[row['ImageId']] = first_prediction


            self.image_names = list(os.listdir(self.images_dir))

        def __len__(self):
            return len(self.image_names)

        def __getitem__(self, idx):

            img_path = os.path.join(self.images_dir, self.image_names[idx])
           # print(img_path)
            image = Image.open(img_path).convert('RGB')

            img_name = os.path.basename(os.path.splitext(self.image_names[idx])[0])

            label_i = self.label_to_index[self.image_name_to_label[img_name]]

            if self.transform:
                image = self.transform(image)

            if self.return_index:
                return image, label_i, idx
            else:
                return image, label_i

class ImageNetTrainDataset(ImageFolder):
    def __init__(
        self,
        root: str,
        imagenet_class_index: str,
        labels_file: str,
        transform: Optional[Callable] = None,
        return_index: bool = True
    ):
        super().__init__(root, transform=transform)
        self.return_index = return_index

        # Load label code to index
        with open(imagenet_class_index, 'r') as file:
            self.label_to_index = {parts[0]: i for i, line in enumerate(file) 
                                   for parts in [line.strip().split()] if parts}

        # Load image name to label code
        with open(labels_file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            self.image_name_to_label = {row['ImageId']: row['PredictionString'].split()[0] 
                                        for row in csv_reader}

        # Create a mapping from class folder names to Kaggle label indices
        self.folder_to_kaggle_index = {}
        for class_name in self.class_to_idx.keys():
            class_path = os.path.join(root, class_name)
            for img_name in os.listdir(class_path):
                img_id = os.path.splitext(img_name)[0]
                if img_id in self.image_name_to_label:
                    label_code = self.image_name_to_label[img_id]
                    self.folder_to_kaggle_index[class_name] = self.label_to_index[label_code]
                    break
            if class_name not in self.folder_to_kaggle_index:
                print(f"Warning: No matching Kaggle label found for class {class_name}. Using original class index.")
                self.folder_to_kaggle_index[class_name] = self.class_to_idx[class_name]

    def __getitem__(self, idx: int) -> Tuple[Any, int, Optional[int]]:
        image, original_class_idx = super().__getitem__(idx)
        class_name = self.classes[original_class_idx]
        kaggle_label_idx = self.folder_to_kaggle_index[class_name]

        if self.return_index:
            return image, kaggle_label_idx, idx
        else:
            return image, kaggle_label_idx


def load_imagenet(imagenet_path, train_test_val, batch_size=32, shuffle=False, drop_last=True, transform=None):

    # assuming the same structure as here: https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description
    imagenet_train_path = os.path.join(imagenet_path, "ILSVRC/Data/CLS-LOC/train")
    imagenet_val_path  =os.path.join(imagenet_path, "ILSVRC/Data/CLS-LOC/val")
    imagenet_train_labels = os.path.join(imagenet_path, "LOC_train_solution.csv")
    imagenet_val_labels = os.path.join(imagenet_path, "LOC_val_solution.csv")
    imagenet_label_strings = os.path.join(imagenet_path, "LOC_synset_mapping.txt" )

    # data_transforms = transforms.Compose([
    #     Resize(224, interpolation=Image.BICUBIC),
    #     CenterCrop(224),
    #     lambda image: image.convert("RGB"),
    #     ToTensor(),
    #     Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    # ])

        
    # Define data transforms
    # Set the appropriate directory based on train_test_val
    if train_test_val == 'train':
        data_dir = imagenet_train_path
        labels = imagenet_train_labels
        dataset = ImageNetTrainDataset(data_dir, imagenet_label_strings, labels, transform=transform)

    elif train_test_val == 'val':
        data_dir = imagenet_val_path
        labels = imagenet_val_labels
        dataset = ImageNetValDataset(data_dir, imagenet_label_strings, labels, transform=transform)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )

    return dataloader

def get_imagenet_names(imagenet_path):
    ind_to_names = {}
    list_of_names = []
    with open(os.path.join(imagenet_path, "LOC_synset_mapping.txt"), 'r') as file:
        for line_num, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            
            # Split the line into synset ID and the rest (names)
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            
            synset = parts[0]
            full_names = parts[1].split(',')
            full_names = [name.strip() for name in full_names if name.strip()]
            
            ind_to_names[line_num] = full_names
            list_of_names.extend(full_names)
    
    return ind_to_names, list_of_names

def get_text_embeddings(model_name, list_of_classes):
    vanilla_model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name, do_rescale=False) # Make sure the do_rescale is false for pytorch datasets
    inputs = processor(text=list_of_classes, return_tensors='pt', padding=True)
    text_embeddings = vanilla_model.get_text_features(**inputs)
    del vanilla_model, processor
    return text_embeddings
