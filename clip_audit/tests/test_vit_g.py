"""
The purpose of this code is to test ViT-G on CC.

"""

from transformers import CLIPProcessor, CLIPModel
import torch

from torch.utils.data import Dataset, DataLoader


import os

from clip_audit.dataloader.conceptual_captions import ConceptualCaptionsDataset


def load_clip_model():
    model_name = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    
    # Load model and processor
    model = CLIPModel.from_pretrained(model_name, cache_dir='/network/scratch/s/sonia.joseph/hub')
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir='/network/scratch/s/sonia.joseph/hub')
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Set to eval mode
    model.eval()
    
    return model, processor

def simple_retrieval_accuracy(model, processor, dataloader, device, num_samples=100):
    correct = 0
    total = 0

    print("starting simple retrieval accuracy task")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images = batch['image'].to(device)
            captions = batch['caption']
            
            # Process text only
            text_inputs = processor.tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
            
            # Forward pass
            outputs = model(
                pixel_values=images,
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
            
            # Get similarity scores
            logits_per_image = outputs.logits_per_image
            
            # Check if highest similarity is on the diagonal
            predicted = logits_per_image.argmax(dim=1)
            correct += (predicted == torch.arange(len(predicted)).to(device)).sum().item()
            total += len(predicted)
    
    accuracy = 100 * correct / total
    print(f"Retrieval Accuracy: {accuracy:.2f}%")
    return accuracy

# Load the model
model, processor = load_clip_model()

# Now your transforms should use the processor
def get_transforms():
    return processor

transforms = get_transforms()
print(transforms)


image_dir = '/network/datasets/conceptualcaptions/Train'
tsv_file_name = 'GCC-training.tsv'
tsv_path = os.path.join(image_dir, tsv_file_name)

# Use in dataset
dataset = ConceptualCaptionsDataset(
    tsv_file=tsv_path,
    image_root_dir=image_dir,
    transform=None
)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

accuracy = simple_retrieval_accuracy(model, processor, dataloader, device='cuda')
