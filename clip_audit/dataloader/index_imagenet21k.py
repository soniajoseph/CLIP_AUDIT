import tarfile
import json
import os
from tqdm import tqdm
import time
from datetime import datetime

def create_tar_index_fast(tar_path, index_save_path):
    """
    Create an index file mapping class IDs to their positions in the tar file
    with detailed timing information
    """
    index = {}
    start_time = time.time()
    
    print(f"Starting index creation at {datetime.now().strftime('%H:%M:%S')}")
    
    tar = tarfile.open(tar_path, 'r|gz')  # Using streaming mode
    try:
        for member in tqdm(tar):
            if member.name.endswith('.tar'):
                class_id = member.name.split('/')[-1].split('.')[0]
                index[class_id] = {
                    'offset': member.offset_data,
                    'size': member.size
                }
                
                # Save progress every 100 classes
                if len(index) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = len(index) / elapsed
                    print(f"\nProcessed {len(index)} classes in {elapsed:.2f} seconds")
                    print(f"Rate: {rate:.2f} classes/second")
                    
                    with open(index_save_path + '.partial', 'w') as f:
                        json.dump(index, f)
    finally:
        tar.close()
    
    # Save final index
    with open(index_save_path, 'w') as f:
        json.dump(index, f)
    
    total_time = time.time() - start_time
    print(f"\nFinished at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Final count: {len(index)} classes")
    print(f"Average rate: {len(index)/total_time:.2f} classes/second")
    
    return index

tar_path = '/network/datasets/imagenet21k/winter21_whole.tar.gz'
index_save_path = '/network/scratch/s/sonia.joseph/datasets/imagenet21k/index.json'

print(f"Starting to create index for {tar_path}")
create_tar_index_fast(tar_path, index_save_path)