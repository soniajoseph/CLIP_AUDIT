import subprocess

# One-time extraction of just the class file
class_id = 'n12203529'
output_dir = '/network/scratch/s/sonia.joseph/datasets/imagenet21k/extracted_classes'
tar_path = '/network/datasets/imagenet21k/winter21_whole.tar.gz'

# Using tar command directly is much faster than Python's tarfile
cmd = f"tar -xzf {tar_path} --wildcards '{class_id}.tar' -C {output_dir}"
subprocess.run(cmd, shell=True)