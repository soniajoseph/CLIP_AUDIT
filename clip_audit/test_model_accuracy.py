
"""
Test accuracy of TinyCLIP on dataset.

Mainly as sanity check and also making sure we're doing our image normalization properly
"""

import os
from clip_audit.utils.load_imagenet import load_imagenet, get_imagenet_names

import argparse
from transformers import CLIPProcessor, CLIPModel
import torch

from vit_prisma.models.base_vit import HookedViT

from torchvision import transforms


from tqdm.auto import tqdm

def main(imagenet_path, train_val, model_type, model_name, max, k=5):

    ind_to_name, imagenet_names = get_imagenet_names(imagenet_path)

    print(imagenet_names)

    # processor = CLIPProcessor.from_pretrained(model_name, do_rescale=False)
    # image_mean = processor.image_processor.image_mean
    # image_std = processor.image_processor.image_std
    
    transform = transforms.Compose([
            transforms.Resize((224, 224)), # Resize the image
            transforms.ToTensor(), # Convert the image to a tensor
            # transforms.Normalize(mean=image_mean, std=image_std)
        ])

    dataloader = load_imagenet(imagenet_path, train_val, shuffle=True, transform=transform)

    device = 'cuda'


    # Check original model
    if model_type == 'original':
        model = CLIPModel.from_pretrained(model_name)
    elif model_type == 'hooked':
        model = HookedViT.from_pretrained(model_name, is_clip=True, is_timm=False)
        
    # Get text features
    # with torch.no_grad():
    #     vanilla_model = CLIPModel.from_pretrained(model_name)
    #     processor = CLIPProcessor.from_pretrained(model_name, do_rescale=False)
    #     inputs = processor(text=imagenet_names, return_tensors='pt', padding=True)
    #     text_features = vanilla_model.get_text_features(**inputs)
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    #     text_features = text_features.to(device)
    #     del vanilla_model

    # Get text features
    with torch.no_grad():
        vanilla_model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name, do_rescale=False)

        # Create a list of all prompts and a mapping from prompt to class index
        all_prompts = []
        prompt_to_class = []
        for class_idx, names in ind_to_name.items():
            for name in names:
                all_prompts.append(f"a photo of a {name}")
                prompt_to_class.append(class_idx)

        # Convert prompt_to_class to a tensor
        prompt_to_class = torch.tensor(prompt_to_class)

        # Process all prompts in batches to avoid potential memory issues
        batch_size = 256
        text_features_list = []
        for i in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[i:i+batch_size]
            inputs = processor(text=batch_prompts, return_tensors='pt', padding=True)
            batch_text_features = vanilla_model.get_text_features(**inputs)
            batch_text_features = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)
            text_features_list.append(batch_text_features)

        # Concatenate all text features
        text_features = torch.cat(text_features_list, dim=0)
        
        # Move tensors to the appropriate device
        text_features = text_features.to(device)
        prompt_to_class = prompt_to_class.to(device)

        del vanilla_model


    # get processor clip normalziation values
    model.to(device)

    # iterate through
    top_1_correct = 0
    top_k_correct = 0
    total = 0

    

    print("Max is ", max)
    print("All prompts are ", all_prompts)

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            inputs  = processor(images=images, return_tensors="pt", do_resize=True, do_center_crop=True, do_normalize=True, do_rescale=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}

        
            if model_type == 'original':
                image_features = model.get_image_features(**inputs)
            else: 
                image_features = model(images)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute similarity scores for all prompts
            similarity = 100.0 * image_features @ text_features.T

            # For each image, get the maximum similarity for each class
            max_similarity = torch.zeros(similarity.shape[0], len(ind_to_name), device=device)
            for class_idx in range(len(ind_to_name)):
                class_mask = (prompt_to_class == class_idx)
                max_similarity[:, class_idx] = similarity[:, class_mask].max(dim=1).values

            # Apply softmax to get probabilities
            similarity = max_similarity.softmax(dim=-1)

            # Get top-k predictions
            _, top_k_indices = torch.topk(similarity, k, dim=-1)

            # Compute top-1 and top-k accuracy
            top_1_correct += (top_k_indices[:, 0] == labels).sum().item()
            top_k_correct += sum([label in pred for label, pred in zip(labels, top_k_indices)])
            total += labels.size(0)

            if max != -1 and total >= max:
                break

        top_1_accuracy = 100 * top_1_correct / total
        top_k_accuracy = 100 * top_k_correct / total
        print(f"Top-1 Accuracy on ImageNet: {top_1_accuracy:.2f}%")
        print(f"Top-{k} Accuracy on ImageNet: {top_k_accuracy:.2f}%")
                    
    #         image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    #         # Compute similarity scores
    #         similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    #         # Get top-k predictions
    #         _, top_k_indices = torch.topk(similarity, k, dim=-1)

    #         # Check if true label is in top-k predictions
    #         top_k_correct += sum([label in pred for label, pred in zip(labels, top_k_indices)])
    #         total += labels.size(0)

    #         if max != -1 and total >= max:
    #             break

    # top_k_accuracy = 100 * top_k_correct / total
    # print(f"Top-{k} Accuracy on ImageNet: {top_k_accuracy:.2f}%")
    # return top_k_accuracy


# Results for old preprocessing (no center crop) 
# Vanilla wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M for 50k ImageNetV Val (top 1): 
# Hooked wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M for 50k ImageNetV Val (top 1): 25.05%

# Vanilla wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M for 50k ImageNetV Val (top 1): 36.35%
# Hooked wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M for 50k ImageNetV Val (top 1): 36.35%

# Hooked wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M for 50k ImageNetV Val (top 1): 38.09%

# openai/clip-vit-base-patch32 for 50k ImageNetV Val (top 1): 35.25% 
# openai/clip-vit-large-patch14 for 50k ImageNetV Val (top 1) 44.42%

# Note: I think there's something wrong with the accuracy function. The values are ~half of what they should be?
# I'm not using center crop in the preprocessing. Check center crop and do that. You want to make sure the activations you're collecting are in-distribution.

# Results for old preprocessing (with center crop):
# Vanilla wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M for 50k ImageNet Val (top 1):  36.36%


# Results for "openai/clip-vit-base-patch32"
# Vanilla openai/clip-vit-base-patch32 for 50k ImageNet Val (top 1):  52.15%

# What now I''m getting different results
# Loaded pretrained model wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M into HookedTransformer
# Evaluating:   2%|██▋                                                                                                                                     | 31/1562 [00:36<30:04,  1.18s/it]
# Top-5 Accuracy on ImageNet: 50.88%

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Test accuracy of TinyCLIP on dataset')
    parser.add_argument('--imagenet_path', type=str, default='/network/scratch/s/sonia.joseph/datasets/kaggle_datasets', help='Path to dataset')
    parser.add_argument('--train_val', type=str, default='val', help='Train, test or validation set')
    parser.add_argument('--model_type', type=str, default='original', help="Model to test (original/hooked)", choices=['original', 'hooked'])
    parser.add_argument('--model_name', type=str, default='wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M', help='Model name to test')
    parser.add_argument('--max', type=int, default=1000, help='Number of images to test on. Put -1 to do full set')
    parser.add_argument('--k', type=int, default=1, help='Top-k accuracy')

    args = parser.parse_args()
    
    main(args.imagenet_path, args.train_val, args.model_type, args.model_name, args.max, args.k)