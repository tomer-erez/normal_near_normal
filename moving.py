import os
import tarfile
import shutil

# Files/folders to include
files_to_tar = [
    "",
    "",
    "",
]

# Where to put the resulting tar
destination = "/home/tomer/backups"

# Name of the tar file
tar_name = "backup.tar.gz"

# Create destination directory if it doesn't exist
os.makedirs(destination, exist_ok=True)

tar_path = os.path.join(destination, tar_name)

# Create tar.gz
with tarfile.open(tar_path, "w:gz") as tar:
    for path in files_to_tar:
        if os.path.exists(path):
            # arcname keeps the path inside the archive clean
            tar.add(path, arcname=os.path.basename(path))
        else:
            print(f"Warning: {path} does not exist")

print(f"Created: {tar_path}")