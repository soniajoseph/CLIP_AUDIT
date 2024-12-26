import h5py
from clip_audit.dataloader.imagenet21k_dataloader_simple_iterator import StreamingImageNet21k
import matplotlib.pyplot as plt

def print_image_ids_hdf5(file_path, start=0, limit=10):
    """
    Print image IDs from HDF5 file
    
    Args:
        file_path: Path to the HDF5 file
        start: Starting index (default 0)
        limit: Maximum number of IDs to print (default 10)
    """
    with h5py.File(file_path, 'r') as f:
        total = len(f['image_ids'])
        print(f"Total image IDs: {total}")
        print("\nSample of image IDs:")
        end = min(start + limit, total)
        for i in range(start, end):
            print(f"Index {i}: {f['image_ids'][i]}")


def get_images_by_class_id(tar_path, ids_file_path, class_id):
    """
    Get all images for a specific class ID (n_ part)
    
    Args:
        tar_path: Path to the main ImageNet21k tar.gz file
        ids_file_path: Path to saved image IDs file
        class_id: The n_ part of the image ID (e.g., 'n02689434')
    
    Returns:
        List of matching image IDs
    """
    matching_ids = []
    with h5py.File(ids_file_path, 'r') as f:
        for idx, image_id_bytes in enumerate(f['image_ids']):
            image_id = image_id_bytes.decode('utf-8')
            if image_id.startswith(class_id):
                matching_ids.append(image_id)
    return matching_ids

def get_image_from_class(tar_path, ids_file_path, class_id, image_index=0):
    """
    Get a specific image from a class
    
    Args:
        tar_path: Path to the main ImageNet21k tar.gz file
        ids_file_path: Path to saved image IDs file
        class_id: The n_ part of the image ID (e.g., 'n02689434')
        image_index: Which image to get from this class (0-based index)
    """
    matching_ids = get_images_by_class_id(tar_path, ids_file_path, class_id)
    if not matching_ids:
        raise ValueError(f"No images found for class {class_id}")
    
    if image_index >= len(matching_ids):
        raise ValueError(f"Image index {image_index} out of range. Class {class_id} has {len(matching_ids)} images")
    
    image_id = matching_ids[image_index]
    dataset = StreamingImageNet21k(tar_path)
    return dataset.find_image(image_id)
    
file_path = '/network/scratch/s/sonia.joseph/CLIP_AUDIT/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/image_ids.h5'
tar_path = '/network/datasets/imagenet21k/winter21_whole.tar.gz'

index ='n02689434_4489'
print_image_ids_hdf5(file_path)


# Get the first image from class 'n02689434'
try:
    result = get_image_from_class(tar_path, file_path, 'n02689434', image_index=0)
    
    image = result['image']
    print(f"Class ID: {result['class_id']}")
    print(f"Image ID: {result['image_id']}")
    
    # Display if in notebook
    plt.figure()
    plt.imshow(image)
    # Or save
    plt.savefig('retrieved_image.jpg')
    
    # Print all available images for this class
    matching_ids = get_images_by_class_id(tar_path, file_path, 'n02689434')
    print(f"\nFound {len(matching_ids)} images for class n02689434:")
    for i, img_id in enumerate(matching_ids):
        print(f"Image {i}: {img_id}")
        
except Exception as e:
    print(f"Error: {str(e)}")