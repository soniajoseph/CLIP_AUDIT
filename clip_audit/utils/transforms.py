import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

def _convert_to_rgb(image):
    return image.convert('RGB')

def get_clip_transform():
    return transforms.Compose([
        transforms.Resize(size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True),
        transforms.CenterCrop(size=(224, 224)),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])


