from vit_prisma.models.base_vit import HookedViT



# model = HookedViT.from_pretrained('laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K', is_clip=True, is_timm=False)
# print(model)

import open_clip

model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K')
tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K')

print(model)