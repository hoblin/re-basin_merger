import os
import subprocess

# Read the models.txt file and create an array of file paths
with open('models.txt', 'r') as f:
    file_paths = [line.strip() for line in f]

# Check if there are at least two files to merge
if len(file_paths) < 2:
    print("Need at least two files to merge.")
    exit(1)

# Initial output file name
output_file = "output-0.0.1"

# Run the initial merge
subprocess.run(f"python SD_rebasin_merge.py --model_a {file_paths[0]} --model_b {file_paths[1]} --output {output_file} --alpha 0.3 --device cuda --iterations 250 --fast --usefp16", shell=True)

# Delete the first two file paths from the list
del file_paths[0:2]

# For each remaining file path in the array
for i, file_path in enumerate(file_paths, start=1):
    # Create the new output file name
    new_output_file = f"output-0.0.{i+1}"

    # Run the merge script with the last output file and the next file path
    subprocess.run(f"python SD_rebasin_merge.py --model_a {output_file}.safetensors --model_b {file_path} --output {new_output_file} --alpha 0.3 --device cuda --iterations 250 --fast --usefp16", shell=True)

    # Delete the old output file
    os.remove(f"{output_file}.safetensors")

    # Update the output file name for the next iteration
    output_file = new_output_file
