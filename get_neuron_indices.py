
from vit_prisma.models.base_vit import HookedViT
import argparse




def load_model(model_name):
    """
    Load model
    """
    model = HookedViT.from_pretrained(model_name, is_timm=False, is_clip=True)
    return model


def main(model_name):
    """
    Main function
    """
    model = load_model(model_name)

    print(model)



# main function
if __name__ == "__main__":

    # activation parser, model name
    parser = argparse.ArgumentParser(description='Get neuron indices')
    parser.add_argument('--model_name', type=str, default="wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M", help='Model name')
    args = parser.parse_args()

    # main function
    main(args.model_name)

