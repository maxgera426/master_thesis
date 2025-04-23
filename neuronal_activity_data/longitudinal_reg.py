import pandas as pd
import numpy as np
import os
import csv


def load_data(file_path):
    print("Processing file: ", file_path)
    correspondences_path = file_path.split(".")[0] + "_Correspondences.csv"
    correspondences = pd.read_csv(correspondences_path)
    return correspondences

def save_data_behavior(output_file, dict):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Cell', 'Sequence', 'Moving To Zone 1', 'Moving To Trough', 'Drinking Full', 'Moving To Zone 2', 'Moving To Lever', 'Drinking Empty', 'Off Task']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key, values in dict.items():
            if int(key) < 10:
                cell_name = f"C0{key}"
            else :
                cell_name = f'C{key}'
            for v in values:
                row_dict = {
                    'Cell': cell_name,
                    'Sequence': v[0],
                    'Moving To Zone 1': v[1],
                    'Moving To Trough': v[2],
                    'Drinking Full': v[3],
                    'Moving To Zone 2': v[4],
                    'Moving To Lever': v[5],
                    'Drinking Empty': v[6],
                    'Off Task': v[7]
                }
                writer.writerow(row_dict)

def save_data_movement(output_file, dict):
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Cell', 'Moving', 'Not Moving']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for key, values in dict.items():
            if int(key) < 10:
                cell_name = f"C0{key}"
            else :
                cell_name = f'C{key}'
            for v in values:
                row_dict = {
                    'Cell': cell_name,
                    'Moving': v[0],
                    'Not Moving': v[1]
                }
                writer.writerow(row_dict)

def fetch_percentiles(correspondences):
    percentiles = {}

    for _, row in correspondences.iterrows():
        cell_id = int(row["global_cell_index"])
        corresponding_id = str(int(row["local_cell_index"]))
        corresponding_session = int(row["local_cellset_index"])
        percentile_file = percentile_files[corresponding_session+1]
        percentile_data = pd.read_csv(percentile_file)
        cell_list = list(percentile_data["Cell"].values)

        possible_cell_names = [f'C{corresponding_id}', f'C0{corresponding_id}', f'C00{corresponding_id}']
        for id in possible_cell_names:
            for name in cell_list:
                print(cell_id, id, name)
                if name == id:
                    percentile_list = list(percentile_data[percentile_data["Cell"] == name].values)[0][1:]
                    if cell_id not in percentiles.keys():
                        percentiles[cell_id] = []
                        percentiles[cell_id].append(percentile_list)
                    else : 
                        percentiles[cell_id].append(percentile_list)
                    break
    
    return percentiles


file_path = r"P:\Ca2+ Data\M15 - Jun24\M15_longitudinal_reward.csv"
percentile_dir = r"neuronal_activity_data\percentiles\M15\movement_related\\"
percentile_files = [percentile_dir + f for f in os.listdir(percentile_dir)]
correspondences = load_data(file_path)
counts = correspondences["global_cell_index"].value_counts()
indices = counts[counts>1].index
correspondences = correspondences[correspondences["global_cell_index"].isin(indices)]

percentiles = fetch_percentiles(correspondences)
save_data_movement(r"neuronal_activity_data\percentiles\M15\movement_related\longitudinal.csv", percentiles)
