import os
import subprocess
import yaml
import requests
import webuiapi

# Constants
MODELS_DIR = "/workspace/stable-diffusion-webui/models/Stable-diffusion/"
ALPHA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
API_HOST = "https://hoblin.ngrok.io"
MERGE_PLAN_API = f"{API_HOST}/admin/checkpoint_merges/1/merge_plan"
UPDATE_ALPHA_API = f"{API_HOST}/admin/merge_steps/{{step_id}}/update_alpha"
HOST = "localhost"
PORT = 3000
ITERATIONS = 250
SEEDS = "3124739301,3124739317,3124739308,3124739318"

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

# Create webui API client
api = webuiapi.WebUIApi(host=HOST, port=PORT)

# Read the prompt parameters from prompt.yaml
with open('prompt.yaml', 'r') as f:
    prompt_params = yaml.safe_load(f)

# Get the merge plan from the merger planner
response = requests.get(MERGE_PLAN_API)
merge_plan = response.json()

# Check if there are at least two files to merge
if len(merge_plan) < 2:
    print("Need at least two files to merge.")
    exit(1)

# For each remaining file path in the array
for i, step in enumerate(merge_plan[1:], start=1):
    # Create an array to store the output file names with the file path of the first model without the extension
    step_id, version_filename, step_alpha = step['step_id'], step['version_filename'], step['alpha']

    first_file_name = merge_plan[0]['version_filename']
    stage_merges = [first_file_name]

    model_a_file_path = os.path.join(MODELS_DIR, first_file_name)
    model_b_file_path = os.path.join(MODELS_DIR, version_filename)

    # if step alpha is not null or empty string use [step['alpha']] else use ALPHA_VALUES
    alpha_list = [float(step_alpha)] if step_alpha else ALPHA_VALUES
    # For each alpha value
    for j, alpha in enumerate(alpha_list, start=1):
        # Create the new output file name
        output_file = f"output-{i}.{j}"

        # Run the merge script with the last output file and the next file path
        subprocess.run(
            f"python SD_rebasin_merge.py --model_a {model_a_file_path} --model_b {model_b_file_path} --output {output_file} --alpha {alpha} --device cuda --iterations {ITERATIONS} --fast --usefp16", shell=True)

        new_model_file_name = f"{output_file}.safetensors"

        # Move the output file to the models directory
        os.rename(new_model_file_name, os.path.join(
            MODELS_DIR, new_model_file_name))

        stage_merges.append(output_file)

    stage_merges.append(version_filename)

    # If there is one alpha value, don't ask the user to choose
    if len(alpha_list) == 1:
        # if the alpha value is zero, skip the step
        if alpha_list[0] == 0.0:
            continue

        merge_plan[0]['version_filename'] = f"output-{i}.1.safetensors"
        continue

    script_args = [
        XYZPlotAvailableTxt2ImgScripts.index("Seed"),
        SEEDS,
        SEEDS,  # x_values_dropdown
        XYZPlotAvailableTxt2ImgScripts.index("Checkpoint name"),
        stage_merges,
        stage_merges,  # y_values_dropdown
        XYZPlotAvailableTxt2ImgScripts.index("Nothing"),
        "",  # ZAxisValues
        "",  # ZAxisValuesDropdown
        "True",  # drawLegend
        "False",  # includeLoneImages
        "False",  # includeSubGrids
        "False",  # noFixedSeeds
        20,  # marginSize
    ]

    # Update models
    api.refresh_checkpoints()

    # Generate a grid of images using the merged model
    prompt_params['script_name'] = "X/Y/Z Plot"
    prompt_params['script_args'] = script_args

    result = api.txt2img(**prompt_params)
    api.util_wait_for_ready()

    # Store the response image
    result.image.save(f"stage-{i}-{version_filename}.png")

    # Ask for user input to select the best model
    chosen_alpha = int(input(
        "Enter the number of the chosen alpha value (1 for 0.1, 2 for 0.3, etc.): "))
    chosen_output_file = f"output-{i}.{chosen_alpha}.safetensors"


    # Delete the unchosen models
    for j, alpha in enumerate(ALPHA_VALUES, start=1):
        if j != chosen_alpha:
            os.remove(os.path.join(MODELS_DIR, f"output-{i}.{j}.safetensors"))

    # if user input is zero, skip the step
    if chosen_alpha == 0:
        # save the alpha value to the database
        requests.patch(UPDATE_ALPHA_API.format(step_id=step_id), params={'alpha': 0.0})
        continue

    # save the alpha value to the database
    requests.patch(UPDATE_ALPHA_API.format(step_id=step_id), params={'alpha': alpha_list[chosen_alpha - 1]})

    # Update the file path for the next iteration
    merge_plan[0]['version_filename'] = chosen_output_file
