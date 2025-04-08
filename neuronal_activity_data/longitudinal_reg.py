import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import random


def load_data_from_serv(file_path):
    print("Processing file: ", file_path)
    props_data = pd.read_csv(file_path)
    props_data = props_data[props_data["Status"]=="accepted"][["Name","CentroidX", 'CentroidY', "Size"]]
    return props_data

def load_local_data(file_path):
    props_data = pd.read_csv(file_path)
    return props_data

def plot_cells(position_data):
    plt.figure()
    for x, y, size in zip(position_data['CentroidX'], position_data['CentroidY'], position_data["Size"]):
        color = (random.random(), random.random(), random.random())
        plt.gca().add_patch(Circle((x,y), size/2, color= color, alpha = 1.0))

    plt.xlim(0, 300)
    plt.ylim(0, 200)
    plt.axis('equal')
    plt.show()

def get_file_list(exp_list, folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("CellTraces-props.csv"):
                if any(f"{exp}" in root for exp in exp_list):
                    file_list.append(os.path.join(root, file))
    return file_list

def compute_distance(pop1, pop2):
    distances = []
    for _, row1 in pop1.iterrows():
        name1, x1, y1, size1 = row1
        pos1 = np.array((x1, y1))
        for _, row2 in pop2.iterrows():
            name2, x2, y2, size2 = row2
            pos2 = np.array((x2, y2))
            dist = np.linalg.norm(pos1-pos2)
            if dist:
                distances.append(dist)
            # print(f"Distance btw {name1} and {name2}: ", dist)
    return distances

def distance_map(list_distances, max_dist, unit):
    distance_map = np.zeros((int(max_dist)+1)*10)
    for cell_pair_distances in list_distances:
        for dist in cell_pair_distances:
            distance_map[int(dist*unit*10)] += 1
    return distance_map

dirs = [r"neuronal_activity_data\cell_props\\" + name for name in ["M2"]]

list_distances = []
for dir in dirs:
    file_list = [os.path.join(dir,f) for f in os.listdir(dir)]
    for i in range(len(file_list)-1):
        for j in range(i+1, len(file_list)):
            file1 = file_list[i]
            file2 = file_list[j]
            pop1 = load_local_data(file1)
            pop2 = load_local_data(file2)
            distances = compute_distance(pop1, pop2)

            list_distances.append(distances)
# plot_cells(load_local_data(r"neuronal_activity_data\cell_props\M2\Exp 010_M2_240619_FR1_1_CellTraces-props.csv"))
p1 = np.array((0,0))
p2 = np.array((200, 302)) # M2 = 302 x 200 pixels --- M4 = 307 x 200 pixels --- M15 = 292 x 200 pixels
pixel_to_µm = 1 # M2 = 3.656 µm/pixel --- M4 = 3.458 µm/pixel --- M15 = 3.356 µm/pixel
max_dist = 300 #np.linalg.norm(p1-p2)*pixel_to_μm
distance_map = distance_map(list_distances, max_dist, pixel_to_μm)
print(np.sum(distance_map))
plt.figure()
plt.bar(np.arange((max_dist +1 )*10)/10, distance_map, width=0.08)
plt.show()