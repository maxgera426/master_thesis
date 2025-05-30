import matplotlib.pyplot as plt
import pandas as pd
import os

mice = ["M2", "M4", "M15"]
props_folder = r"neuronal_activity_data\cell_props"

FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, mouse in enumerate(mice):

    folder = os.path.join(props_folder, mouse)

    FR1_file_list = [os.path.join(folder, f) for f in os.listdir(folder) 
                     if any(exp in f for exp in FR1)]
 
    all_points = pd.DataFrame(columns=["CentroidX", "CentroidY"])

    for f in FR1_file_list:
        data = pd.read_csv(f)[["CentroidX", "CentroidY"]]
        all_points = pd.concat([all_points, data])
    
    # Plot on current subplot
    ax = axes[i]
    ax.set_xlim(0, 320)
    ax.set_ylim(0, 200)
    
    ax.plot(all_points['CentroidX'], all_points['CentroidY'], 'ro', markersize=4)
    
    # Customize each subplot
    ax.set_xlabel('X Coordinate (pixels)')
    if i == 0:  # Only add Y label to first subplot to avoid repetition
        ax.set_ylabel('Y Coordinate (pixels)')
    ax.set_title(f'{mouse} (n={len(all_points)} cells)')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
