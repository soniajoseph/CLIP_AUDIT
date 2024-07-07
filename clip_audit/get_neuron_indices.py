
from vit_prisma.models.base_vit import HookedViT
import argparse

from collections import OrderedDict

import numpy as np


def main(save_file_name, total_neurons, n, layers):
    """
    Main function
    """

    # Dictionary of indices per layer
    indices_dict = OrderedDict()

    for l in range(layers):
        # Generate n random numbers out of total number of neurons, no replacement
        indices_dict[l] = np.random.choice(total_neurons, n, replace=False)

    print(indices_dict)

    # Save dictionary to folder
    np.save(save_file_name, indices_dict)

    print(f"Saved neuron indices to {save_file_name}")

    return indices_dict


# main function
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get neuron indices")
    parser.add_argument("--save_file_name", default="./saved_data/tinyclip_neuron_indices_mlp_out.npy", type=str, help="Model name")
    parser.add_argument("--total_neurons", default=512, type=int, help="Total number of neurons")
    parser.add_argument("--n", default=100, type=int, help="Number of neurons to sample")
    parser.add_argument("--layers", default=12, type=int, help="Number of layers")
    args = parser.parse_args()

    main(args.save_file_name, args.total_neurons, args.n, args.layers)
