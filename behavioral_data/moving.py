import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import os

def plot_description_and_movements(behavior_file, movement_file):

    behavior_data = pd.read_csv(behavior_file)
    movement_data = pd.read_csv(movement_file)
    
    fig = plt.figure(figsize=(12, 4))

    start_pressing_times = behavior_data['Pressing Start'].dropna()
    end_pressing_times = behavior_data['Pressing End'].dropna()
    full_licking_times_start= behavior_data['Licking Full Start'].dropna()
    full_licking_times_end= behavior_data['Licking Full End'].dropna()
    empty_licking_times_start = behavior_data['Licking Empty Start'].dropna()
    empty_licking_times_end = behavior_data['Licking Empty End'].dropna()
    start_in_lever_task = behavior_data['In Lever Task Start'].dropna()
    end_in_lever_task = behavior_data['In Lever Task End'].dropna()
    start_in_trough_task = behavior_data['In Trough Task Start'].dropna()
    end_in_trough_task = behavior_data['In Trough Task End'].dropna()
    start_off_task = behavior_data['Off Task Start'].dropna()
    end_off_task = behavior_data['Off Task End'].dropna()
    start_drinking_times = behavior_data['Drinking Full Start']
    end_drinking_times = behavior_data['Drinking Full End']
    start_empty_drinking_df = behavior_data['Drinking Empty Start'].dropna()
    end_empty_drinking_df = behavior_data['Drinking Empty End'].dropna()
    seq_start_times = behavior_data['Sequence Start'].dropna()
    seq_end_times = behavior_data['Sequence End'].dropna()
    start_moving_to_lever = behavior_data["Moving To Lever Start"].dropna()
    end_moving_to_lever = behavior_data["Moving To Lever End"].dropna()
    start_moving_to_trough = behavior_data["Moving To Trough Start"].dropna()
    end_moving_to_trough = behavior_data["Moving To Trough End"].dropna()
    enter_zone1_times = behavior_data["Enter Zone 1"].dropna()
    enter_zone2_times = behavior_data["Enter Zone 2"].dropna()

    plt.barh(y=1.25, width=np.array(end_pressing_times) - np.array(start_pressing_times), left=start_pressing_times, height=0.5, color="black", edgecolor='black', label='Press')
    
    plt.barh(y=.125, width=np.array(end_in_lever_task) - np.array(start_in_lever_task), left=start_in_lever_task, height=0.25, color="red", edgecolor='black', label='In lever task')
    plt.barh(y=.125, width=np.array(end_in_trough_task) - np.array(start_in_trough_task), left=start_in_trough_task, height=0.25, color="blue", edgecolor='black', label='In trough task')
    plt.barh(y=.125, width=np.array(end_off_task) - np.array(start_off_task), left=start_off_task, height=0.25, color="gray", edgecolor='black', label='Off task')
    
    plt.barh(y=1.75, width=np.array(end_drinking_times) - np.array(start_drinking_times), left=np.array(start_drinking_times), height=0.5, color="purple", edgecolor='black', label='Drinking full')
    plt.barh(y=1.75, width=np.array(end_empty_drinking_df) - np.array(start_empty_drinking_df), left=np.array(start_empty_drinking_df), height=0.5, color="orange", edgecolor='black', label='Drinking empty')
    
    ax = plt.gca()

    for i in range(len(full_licking_times_start)):
        start = full_licking_times_start[i]
        end = full_licking_times_end[i]
        width = end - start
        # Create an ellipse patch
        ellipse = patches.Ellipse(
            xy=(start + width/2, 2.02),  # Center of ellipse (x, y)
            width=width,  # Width of ellipse
            height=0.02,  # Height of ellipse
            angle=0,  # Angle of ellipse
            facecolor='purple',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(ellipse)

    # Draw empty licks as ovals
    for i in range(len(empty_licking_times_start)):
        start = empty_licking_times_start[i]
        end = empty_licking_times_end[i]
        width = end - start
        # Create an ellipse patch
        ellipse = patches.Ellipse(
            xy=(start + width/2, 2.02),  # Center of ellipse
            width=width,
            height=0.02,
            angle=0,
            facecolor='orange',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(ellipse)

    plt.barh(y=0.75, width=np.array(seq_end_times) - np.array(seq_start_times), left=seq_start_times, height=0.5, color="green", edgecolor='black', label='Action sequence')
    plt.barh(y=.375, width = end_moving_to_lever - start_moving_to_lever, left= start_moving_to_lever, height=0.25, color="darkred", edgecolor='black', label="moving to lever zone")
    plt.barh(y=.375, width = end_moving_to_trough - start_moving_to_trough, left= start_moving_to_trough, height=0.25, color="navy", edgecolor='black', label="moving to trough zone")
    
    ymin = -0.1
    ymax = 2.3
    # plt.vlines(np.array(seq_start_times), ymin, ymax, colors='green', linestyles="dashed", label="Press event")
    # plt.vlines(np.array(seq_end_times), ymin, ymax, colors='green', linestyles="dashed")
    # plt.vlines(np.array(enter_zone1_times), ymin, ymax, colors='blue', linestyles="dashed", label="Zone 1 event")
    # plt.vlines(np.array(enter_zone2_times), ymin, ymax, colors='red', linestyles="dashed", label="Zone 2 event")
    # plt.vlines(np.concatenate((start_drinking_times, end_drinking_times, start_empty_drinking_df, end_empty_drinking_df)), ymin, ymax, colors="black", linestyles="dashed", label="Drinking event")

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
               markeredgecolor='black', markersize=8, label='Full licks'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
               markeredgecolor='black', markersize=8, label='Empty licks')
    ]

    handles, labels = ax.get_legend_handles_labels()

    handles.extend(legend_elements)

    
    for i, segment in movement_data.iterrows():
        if i == 0:
            plt.axvspan(segment[0]*1000, segment[1]*1000, 
                        alpha=0.2, color='green', label="Movement intervals")
        else: 
            plt.axvspan(segment[0]*1000, segment[1]*1000, 
                        alpha=0.2, color='green')

    y_ticks = [0.125, 0.375, 0.75, 1.25, 1.75]
    y_labels = ["Task", "Moving", "Sequence", "Press", "Drinking"]
    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Time (ms)")
    plt.legend(handles=handles, loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_refined_movements(movement_data, lever_intersections, trough_intersections, moving_to_lever, moving_to_trough):
    
    plt.figure(figsize=(12, 2))
    plt.barh(y=.375, width = moving_to_lever.iloc[:,1] - moving_to_lever.iloc[:,0], left= moving_to_lever.iloc[:,0], height=0.24, color="darkred", edgecolor='black', label="moving to lever zone")
    plt.barh(y=.375, width = moving_to_trough.iloc[:,1] - moving_to_trough.iloc[:,0], left= moving_to_trough.iloc[:,0], height=0.24, color="navy", edgecolor='black', label="moving to trough zone")
    plt.barh(y=.625, width = lever_intersections.iloc[:,1] - lever_intersections.iloc[:,0], left= lever_intersections.iloc[:,0], height=0.24, color="orangered", edgecolor='black', label="locomotion to lever zone")
    plt.barh(y=.625, width = trough_intersections.iloc[:,1] - trough_intersections.iloc[:,0], left= trough_intersections.iloc[:,0], height=0.24, color="cornflowerblue", edgecolor='black', label="locomotion to trough zone")
    

    for i, segment in movement_data.iterrows():
        if i == 0:
            plt.axvspan(segment[0]*1000, segment[1]*1000, 
                        alpha=0.2, color='green', label="Movement intervals")
        else: 
            plt.axvspan(segment[0]*1000, segment[1]*1000, 
                        alpha=0.2, color='green')
    
    plt.legend(loc = "upper right")
    y_ticks = [0.375, 0.675]
    y_labels = ["Moving", "Locomotion"]
    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()

def refine_movement(behavior_file, movement_file):
    behavior_data = pd.read_csv(behavior_file)
    movement_data = pd.read_csv(movement_file)

    moving_to_lever = behavior_data[["Moving To Lever Start", "Moving To Lever End"]].dropna()
    moving_to_trough = behavior_data[["Moving To Trough Start", "Moving To Trough End"]].dropna()

    lever_intersections = intersection(movement_data, moving_to_lever)
    trough_intersections = intersection(movement_data, moving_to_trough)
     
    plot_refined_movements(movement_data, lever_intersections, trough_intersections, moving_to_lever, moving_to_trough)

    behavior_data["Locomotion To Lever Start"] = lever_intersections["start"]
    behavior_data["Locomotion To Lever End"] = lever_intersections["end"]
    behavior_data["Locomotion To Trough Start"] = trough_intersections["start"]
    behavior_data["Locomotion To Trough End"] = trough_intersections["end"]

    return behavior_data


def intersection(df1, df2):
    intersections = []
    for i, row_1 in df1.iterrows():
        
        start1 = row_1.iloc[0]*1000
        end1 = row_1.iloc[1]*1000

        for j, row_2 in df2.iterrows():
            start2 = row_2.iloc[0]
            end2 = row_2.iloc[1]

            intersection_start = max(start1, start2)
            intersection_end = min(end1, end2)


            if intersection_start < intersection_end:
                intersections.append([intersection_start, intersection_end])
    
    if intersections:
        return pd.DataFrame(intersections, columns=["start", "end"])
    else:
        return pd.DataFrame(columns=["start", "end"])

    

def main():

    mice = ["M2", "M4", "M15"]
    for mouse in mice :
        save_folder = f"behavioral_data\\behavior descriptions\\final_description\\{mouse}"
        behavior_folder = f"behavioral_data\\behavior descriptions\\complete_description\\{mouse}"
        movement_folder = f"behavioral_data\\behavior descriptions\\movement_description\\{mouse}"
        behavior_list = [os.path.join(behavior_folder, f) for f in os.listdir(behavior_folder)]
        movement_list = [os.path.join(movement_folder, f) for f in os.listdir(movement_folder)]

        for behavior_file, movement_file in zip(behavior_list, movement_list):
            file_name = os.path.basename(behavior_file)
            data = refine_movement(behavior_file, movement_file)
            # data.to_csv(os.path.join(save_folder, file_name))

    return

if __name__ == "__main__" :
    main()