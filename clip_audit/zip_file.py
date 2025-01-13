import os
import zipfile
from tqdm import tqdm

def zip_directory(source_dir, output_zip):
    """
    Zip the contents of source_dir into output_zip with a progress bar.
    
    Args:
        source_dir (str): Path to directory to zip
        output_zip (str): Path for output zip file
    """
    # Get total number of files for progress bar
    total_files = sum([len(files) for _, _, files in os.walk(source_dir)])
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Progress bar setup
        with tqdm(total=total_files, desc="Zipping files") as pbar:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Calculate path inside zip file
                    arcname = os.path.relpath(file_path, source_dir)
                    # Add file to zip
                    zipf.write(file_path, arcname)
                    pbar.update(1)

if __name__ == "__main__":
    source_dir = "/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons"
    output_zip = "/network/scratch/s/sonia.joseph/CLIP_AUDIT/sampled_images/open-clip_laion_CLIP-ViT-B-32-DataComp.XL-s13B-b90K/imagenet21k/train/all_neurons.zip"
    
    print(f"Starting to zip {source_dir}")
    zip_directory(source_dir, output_zip)
    print(f"Finished zipping to {output_zip}")