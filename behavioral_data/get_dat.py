import os
import csv

# Extract all the .dat paths in a specified folder corresponding to a list of experiments. 
# The paths are stored in a specified csv file

def extract_dat_paths(folder_path, output_csv, experiment_nums):
    dat_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dat"):
                if any(f"{num}" in root for num in experiment_nums):
                    dat_files.append(os.path.join(root, file))
    

        
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['File']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for file in dat_files:
            writer.writerow({'File': file})
            


folder_path = r"P:\Ca2+ Data\M2 - Jun24"
output_csv = r"C:\Users\maxge\OneDrive - Université Libre de Bruxelles\MA2\Mémoire\master_thesis\behavioral_data\paths\paths_dat\M2_dat_exp10_to_16.csv"
experiment_nums = [f"0{i}" for i in range(10,18)]

extract_dat_paths(folder_path, output_csv, experiment_nums)
