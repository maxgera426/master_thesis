import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_transition_matrix(file_path):
    data = pd.read_csv(file_path)
    full_cycle_order = ['Sequence ', 'Moving To Zone 1 ', 'Moving To Trough ', 'Drinking Full ', 'Moving To Zone 2 ', 'Moving To Lever ', 'Drinking Empty ', 'Off Task ']
    states = []
    for column_name in data.columns:
        if "Start" in column_name and "Press" not in column_name and "Lever Task" not in column_name and "Trough Task" not in column_name:
            states.append(column_name.split("Start")[0])
    

    state_df = pd.DataFrame(columns=["state", "time"])
    for state in states:
        column = data[state + "Start"].dropna()
        for time in column:
            state_df = state_df._append({"state": state, "time": time}, ignore_index=True)
    state_df = state_df.sort_values(by="time").reset_index(drop=True)

    transition_matrix = np.zeros((len(states), len(states)))
    for i in range(len(state_df) - 1):
        current_state = state_df["state"][i]
        next_state = state_df["state"][i + 1]
        transition_matrix[full_cycle_order.index(current_state)][full_cycle_order.index(next_state)] += 1

    return transition_matrix

def plot_transition_matrix(file_list):
    states = ['Seq.', 'Mov. To Zone 1', 'Mov. Trough', 'Drink. Full', 'Mov. To Zone 2', 'Mov. Lever', 'Drink. Empty', 'Off Task']
    if len(file_list) == 1:
        print(file_list[0])
        transition_matrix = get_transition_matrix(file_list[0])
        plt.figure()
        sns.heatmap(transition_matrix, annot=True, cmap = 'Reds', square=True, xticklabels=states, yticklabels=states)
        plt.ylabel("From behavior")
        plt.xlabel("To behavior")
        plt.title("Transition Matrix")
        plt.tight_layout()
        plt.show()
    else:
        fig, axs = plt.subplots(2, len(file_list)//2)
        for file in file_list:  
            transition_matrix = get_transition_matrix(file)
            sns.heatmap(transition_matrix, annot=True, cmap = 'Reds', square=True, xticklabels=states, yticklabels=states, cbar=False, ax=axs[file_list.index(file)//4][file_list.index(file)%4])
            axs[file_list.index(file)//4][file_list.index(file)%4].set_ylabel("From behavior")
            axs[file_list.index(file)//4][file_list.index(file)%4].set_xlabel("To behavior")
            # axs[file_list.index(file)//4][file_list.index(file)%4].set_title("Transition Matrix")
        plt.title("Transition Matrix")
        plt.tight_layout()
        plt.show()

file_list = [r"behavioral_data\behavior descriptions\full session\M2\\" + f for f in os.listdir(r"behavioral_data\behavior descriptions\full session\M2")]
plot_transition_matrix(file_list[0:1])
