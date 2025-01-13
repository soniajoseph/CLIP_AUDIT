from vit_prisma.models.base_vit import HookedViT
from vit_prisma.configs import HookedViTConfig

import torch

# load old model 
model_name = 'open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K'
old_model = HookedViT.from_pretrained(model_name, is_clip=True, is_timm=False, fold_ln=False).to('cuda')

cfg = old_model.cfg

random_model1 = HookedViT(cfg)
random_model1.init_weights()

random_model2 = HookedViT(cfg)
random_model2.init_weights()

# Check if weights are different
print("Are attention weights identical?")
print(torch.allclose(random_model1.W_Q, random_model2.W_Q))  # Should print False

