import os
import subprocess
import yaml
import webuiapi

# Constants
MODELS_DIR = "/workspace/stable-diffusion-webui/models/Stable-diffusion/"
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
HOST = "localhost"
PORT = 3000
ITERATIONS = 10

XYZPlotAvailableTxt2ImgScripts = [
    "Nothing",
    "Seed",
    "Var. seed",
    "Var. strength",
    "Steps",
    "Hires steps",
    "CFG Scale",
    "Prompt S/R",
    "Prompt order",
    "Sampler",
    "Checkpoint name",
    "Sigma Churn",
    "Sigma min",
    "Sigma max",
    "Sigma noise",
    "Eta",
    "Clip skip",
    "Denoising",
    "Hires upscaler",
    "VAE",
    "Styles",
]

# Create API client and wait for job complete
api = webuiapi.WebUIApi(host=HOST, port=PORT)
api.util_wait_for_ready()

# Read the models.csv file and create an array of file paths
file_paths = []
with open('models.csv', 'r') as f:
    for line in f:
        file_name, _url = line.strip().split(',')
        file_paths.append(os.path.join(MODELS_DIR, file_name))

# Read the prompt parameters from prompt.yaml
with open('prompt.yaml', 'r') as f:
    prompt_params = yaml.safe_load(f)

# Check if there are at least two files to merge
if len(file_paths) < 2:
    print("Need at least two files to merge.")
    exit(1)

# For each remaining file path in the array
for i, file_path in enumerate(file_paths[1:], start=1):
    # For each alpha value
    stage_merges = []
    for j, alpha in enumerate(ALPHA_VALUES, start=1):
        # Create the new output file name
        output_file = f"output-{i}.{j}"

        # Run the merge script with the last output file and the next file path
        subprocess.run(
            f"python SD_rebasin_merge.py --model_a {file_paths[0]} --model_b {file_path} --output {output_file} --alpha {alpha} --device cuda --iterations {ITERATIONS} --fast --usefp16", shell=True)

        new_model_file_name = f"{output_file}.safetensors"

        # Move the output file to the models directory
        os.rename(new_model_file_name, os.path.join(
            MODELS_DIR, new_model_file_name))

        stage_merges.append(new_model_file_name)

    script_args = [
        XYZPlotAvailableTxt2ImgScripts.index("Checkpoint name"),
        [stage_merges],
        [stage_merges],  # x_values_dropdown
        XYZPlotAvailableTxt2ImgScripts.index("Seed"),
        "-1,-1,-1",
        "-1,-1,-1",  # y_values_dropdown
        XYZPlotAvailableTxt2ImgScripts.index("Nothing"),
        "",  # ZAxisValues
        "",  # ZAxisValuesDropdown
        "True",  # drawLegend
        "False",  # includeLoneImages
        "False",  # includeSubGrids
        "False",  # noFixedSeeds
        20,  # marginSize
    ]

    # Generate a grid of images using the merged model
    prompt_params['script_name'] = "X/Y/Z Plot"
    prompt_params['script_args'] = script_args
    result = api.txt2img(**prompt_params)
    api.util_wait_for_ready()

    # Store the response image
    result.image.save(f"{output_file}.png")

    # Ask for user input to select the best model
    chosen_alpha = int(input(
        "Enter the number of the chosen alpha value (1 for 0.1, 2 for 0.3, etc.): "))
    chosen_output_file = f"output-{i}.{chosen_alpha}.safetensors"

    # Delete the unchosen models
    for j, alpha in enumerate(ALPHA_VALUES, start=1):
        if j != chosen_alpha:
            os.remove(os.path.join(MODELS_DIR, f"output-{i}.{j}.safetensors"))
            os.remove(f"output-{i}.{j}.png")

    # Update the file path for the next iteration
    file_paths[0] = os.path.join(MODELS_DIR, chosen_output_file)
