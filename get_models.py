import os
import subprocess
import requests
from pathlib import Path

# Constants
MODELS_DIR = Path("/workspace/stable-diffusion-webui/models/Stable-diffusion/")
API_HOST = "https://hoblin.ngrok.io"
MERGE_PLAN_API = f"{API_HOST}/admin/checkpoint_merges/1/merge_plan"
UPDATE_FILENAME_API = f"{API_HOST}/admin/source_checkpoint_versions/{{version_id}}/update_filename"

def download_model(version_id, url):
    # log the version id and url
    print(f"Downloading model with version id {version_id} from {url}")

    # Download the model file
    subprocess.run(f"wget --content-disposition {url} -P {MODELS_DIR}", shell=True)

    # Get the filename of the most recently downloaded file
    file_name = max(MODELS_DIR.glob('*'), key=os.path.getctime).name

    # log the filename
    print(f"Downloaded model with version id {version_id} to {file_name}")

    # Send PATCH request to update the filename in the Rails app
    requests.patch(UPDATE_FILENAME_API.format(version_id=version_id), params={'file_name': file_name})

# Get the merge plan from the Rails app
response = requests.get(MERGE_PLAN_API)
merge_plan = response.json()

# Download the models
for step in merge_plan:
    download_model(step['version_id'], step['version_download_link'])
