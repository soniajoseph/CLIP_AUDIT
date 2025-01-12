from vit_prisma.models.base_vit import HookedViT
from vit_prisma.configs import HookedViTConfig


# load old model 
model_name = 'open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K'
old_model = HookedViT.from_pretrained(model_name, is_clip=True, is_timm=False, fold_ln=False).to('cuda')

cfg = old_model.cfg

random_modelmodel = HookedViT(cfg)
random_model.init_weights()