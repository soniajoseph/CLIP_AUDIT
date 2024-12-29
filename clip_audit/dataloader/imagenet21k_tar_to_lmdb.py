import lmdb
import tarfile
import io
from tqdm import tqdm
import os
from PIL import Image
import numpy as np

class TarToLMDB:
    def __init__(self, tar_path, lmdb_path, map_size=1.5e12):
        """
        tar_path: path to your tar.gz file
        lmdb_path: where to create the LMDB
        map_size: maximum size of database in bytes
        """
        self.tar_path = tar_path
        self.lmdb_path = lmdb_path
        
        # Create scratch directory if it doesn't exist
        scratch_dir = os.path.dirname(lmdb_path)
        os.makedirs(scratch_dir, exist_ok=True)
        
        # Set environment variables for LMDB to use scratch space
        os.environ['TMPDIR'] = scratch_dir
        
        # Open LMDB with specific settings for large databases
        self.env = lmdb.open(
            lmdb_path,
            map_size=map_size,
            metasync=False,
            sync=True,
            map_async=True,
            writemap=True,  # Use write-map to reduce memory usage
            max_spare_txns=1,
            max_readers=1
        )
        
    def _count_images(self):
        """Count total number of images for progress bar"""
        total = 0
        with tarfile.open(self.tar_path, 'r|gz') as tar:
            for member in tar:
                if member.name.endswith('.tar'):
                    class_tar_data = tar.extractfile(member)
                    class_tar = tarfile.open(fileobj=io.BytesIO(class_tar_data.read()))
                    total += sum(1 for m in class_tar.getmembers() 
                               if m.name.endswith(('.JPEG', '.jpg', '.jpeg', '.png')))
                    class_tar.close()
        return total

    def convert(self):
        """Convert tar.gz to LMDB"""
        total_images = self._count_images()
        print(f"Found {total_images} images to process")
        
        # Track statistics
        stats = {
            'total_size': 0,
            'num_images': 0,
            'classes': set()
        }
        
        with self.env.begin(write=True) as txn:
            with tarfile.open(self.tar_path, 'r|gz') as tar:
                pbar = tqdm(total=total_images, desc="Converting images")
                
                for member in tar:
                    if not member.name.endswith('.tar'):
                        continue
                        
                    # Extract class ID from tar filename
                    class_id = os.path.basename(member.name).split('.')[0]
                    stats['classes'].add(class_id)
                    
                    # Process class tar
                    try:
                        class_tar_data = tar.extractfile(member)
                        if class_tar_data is None:
                            print(f"Warning: Could not extract {member.name}")
                            continue
                            
                        class_tar = tarfile.open(fileobj=io.BytesIO(class_tar_data.read()))
                        
                        # Process each image in class
                        for img_member in class_tar.getmembers():
                            if not img_member.name.endswith(('.JPEG', '.jpg', '.jpeg', '.png')):
                                continue
                                
                            # Extract image ID and data
                            image_id = os.path.splitext(img_member.name)[0]
                            img_data = class_tar.extractfile(img_member)
                            
                            if img_data is None:
                                print(f"Warning: Could not extract {img_member.name}")
                                continue
                            
                            # Create key and store data
                            key = f"{class_id}/{image_id}".encode()
                            value = img_data.read()
                            
                            txn.put(key, value)
                            
                            # Update statistics
                            stats['total_size'] += len(value)
                            stats['num_images'] += 1
                            pbar.update(1)
                            
                            # Commit every 5000 images to avoid memory issues
                            if stats['num_images'] % 5000 == 0:
                                txn.commit()
                                txn = self.env.begin(write=True)
                        
                        class_tar.close()
                        
                    except Exception as e:
                        print(f"Error processing {member.name}: {e}")
                        continue
                        
                pbar.close()
        
        # Store metadata
        with self.env.begin(write=True) as txn:
            metadata = {
                'num_images': stats['num_images'],
                'total_size': stats['total_size'],
                'num_classes': len(stats['classes']),
                'classes': list(stats['classes'])
            }
            txn.put(b'__metadata__', str(metadata).encode())
        
        print(f"\nConversion complete!")
        print(f"Total images: {stats['num_images']}")
        print(f"Total size: {stats['total_size'] / (1024*1024*1024):.2f} GB")
        print(f"Number of classes: {len(stats['classes'])}")
        
    def close(self):
        """Close the LMDB environment"""
        self.env.close()

class LMDBImageNet:
    """Class to read from the created LMDB"""
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
        
        # Load metadata
        with self.env.begin() as txn:
            metadata = eval(txn.get(b'__metadata__').decode())
            self.num_images = metadata['num_images']
            self.num_classes = metadata['num_classes']
            self.classes = metadata['classes']
    
    def find_image(self, class_id, image_id, transform=None):
        """Find and load an image by class_id and image_id"""
        key = f"{class_id}/{image_id}".encode()
        
        with self.env.begin() as txn:
            img_bytes = txn.get(key)
            if img_bytes is None:
                return None
            
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            if transform:
                img = transform(img)
                
            return {
                'image': img,
                'class_id': class_id,
                'image_id': image_id
            }
    
    def close(self):
        self.env.close()

# Usage example:
if __name__ == "__main__":
    # Convert tar.gz to LMDB
    lmdb_path='/network/scratch/s/sonia.joseph/datasets/imagenet21k/imagenet21k_db'
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    converter = TarToLMDB(
        tar_path='/network/datasets/imagenet21k/winter21_whole.tar.gz',
        lmdb_path = lmdb_path, # 1.5TB map size
        map_size=1.5e12,
    )
    converter.convert()
    converter.close()
    
    # Test reading from LMDB
    db = LMDBImageNet(lmdb_path)
    print(f"Database contains {db.num_images} images in {db.num_classes} classes")
    
    # Example lookup
    result = db.find_image('class_id', 'image_id')
    if result:
        print("Found image!")
        result['image'].show()
    
    db.close()