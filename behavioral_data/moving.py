import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_description_and_movements(behavior_file, movement_file):

    behavior_data = pd.read_csv(behavior_file)
    movement_data = pd.read_csv(movement_file)
    
    fig = plt.figure(figsize=(12, 4))

    start_pressing_times = behavior_data['Pressing Start'].dropna()
    end_pressing_times = behavior_data['Pressing End'].dropna()
    full_drinking_times = behavior_data['Licking full'].dropna()
    empty_drinking_times = behavior_data['Licking Empty'].dropna()
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
    y_empty_drinking = [1.75]*len(empty_drinking_times)
    y_full_drinking = [1.75]*len(full_drinking_times)
    plt.scatter(full_drinking_times,y_full_drinking, edgecolor='black',color='purple', label = "Full licks")
    plt.scatter(empty_drinking_times,y_empty_drinking, edgecolor='black',color='orange', label = "Empty licks")
    plt.barh(y=0.75, width=np.array(seq_end_times) - np.array(seq_start_times), left=seq_start_times, height=0.5, color="green", edgecolor='black', label='Action sequence')
    plt.barh(y=.375, width = end_moving_to_lever - start_moving_to_lever, left= start_moving_to_lever, height=0.25, color="darkred", edgecolor='black', label="moving to lever")
    plt.barh(y=.375, width = end_moving_to_trough - start_moving_to_trough, left= start_moving_to_trough, height=0.25, color="navy", edgecolor='black', label="moving to trough")
    ymin = -0.1
    ymax = 2
    # plt.vlines(np.array(seq_start_times), ymin, ymax, colors='green', linestyles="dashed", label="Press event")
    # plt.vlines(np.array(seq_end_times), ymin, ymax, colors='green', linestyles="dashed")
    # plt.vlines(np.array(enter_zone1_times), ymin, ymax, colors='navy', linestyles="dashed", label="Zone 1 event")
    # plt.vlines(np.array(enter_zone2_times), ymin, ymax, colors='darkred', linestyles="dashed", label="Zone 2 event")
    # plt.vlines(np.concatenate((start_drinking_times, end_drinking_times, start_empty_drinking_df, end_empty_drinking_df)), ymin, ymax, colors="black", linestyles="dashed", label="Drinking event")

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
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def main():
    behavior_file = r"behavioral_data\behavior descriptions\final_description\M2\M2 - Jun24_Exp 010_behavior_description.csv"
    movement_file = r"behavioral_data\behavior descriptions\movement_description\M2\Exp 010_movement_segments.csv"

    plot_description_and_movements(behavior_file, movement_file)
    return

if __name__ == "__main__" :
    main()