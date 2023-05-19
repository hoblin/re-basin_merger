import os
import subprocess

# Constants
MODELS_DIR = "/workspace/stable-diffusion-webui/models/Stable-diffusion/"

def download_model(file_name, url):
    # Check if the model file exists
    if not os.path.exists(os.path.join(MODELS_DIR, file_name)):
        # Download the model file
        if "civitai.com" in url:
            subprocess.run(f"wget --content-disposition {url} -P {MODELS_DIR}", shell=True)
        elif "drive.google.com" in url:
            file_id = url.split('/')[-2]
            subprocess.run(f"gdown --id {file_id} --output {os.path.join(MODELS_DIR, file_name)}", shell=True)

# Read the models.csv file
with open('models.csv', 'r') as f:
    for line in f:
        file_name, url = line.strip().split(',')
        download_model(file_name, url)
