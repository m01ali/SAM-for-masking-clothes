import os
import requests
from tqdm import tqdm

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

# Create directory for checkpoints if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)

# Download SAM ViT-H checkpoint
checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
checkpoint_path = "checkpoints/sam_vit_h_4b8939.pth"

if not os.path.exists(checkpoint_path):
    print(f"Downloading SAM checkpoint to {checkpoint_path}...")
    download_file(checkpoint_url, checkpoint_path)
    print("Download complete!")
else:
    print(f"SAM checkpoint already exists at {checkpoint_path}")