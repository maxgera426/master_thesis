import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from pycirclize import Circos
from markov_chain.markovchain import MarkovChain
from matplotlib.colors import Normalize


def get_transition_matrix(file_path, full_cycle_order):
    data = pd.read_csv(file_path)
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

def heatmap(matrices, save_folder = None, exp_num = None):
    states = ['Seq.', 'Mov. Trough', 'Drink. Full', 'Mov. Lever', 'Drink. Empty', 'Off Task']
    save_file = os.path.join(save_folder, exp_num + "_heatmap.png")
    if len(matrices) == 1:
        transition_matrix = matrices[0]
        plt.figure()
        annot_labels = np.array([[str(round(val, 1)) for val in row] for row in transition_matrix])
        sns.heatmap(transition_matrix, annot=annot_labels, fmt="", cmap='Blues', square=True, 
                   xticklabels=states, yticklabels=states)
        plt.ylabel("From behavior")
        plt.xlabel("To behavior")
        plt.title("Transition Matrix")
        plt.tight_layout()
        plt.savefig(save_file)
        plt.show()
    else:
        fig, axs = plt.subplots(2, len(matrices)//2)
        for transition_matrix in matrices:  
            annot_labels = np.array([[str(int(val)) for val in row] for row in transition_matrix])
            sns.heatmap(transition_matrix, annot=annot_labels, fmt="", cmap='Blues', square=True, 
                   xticklabels=states, yticklabels=states)            
            axs[matrices.index(transition_matrix)//4][matrices.index(transition_matrix)%4].set_ylabel("From behavior")
            axs[matrices.index(transition_matrix)//4][matrices.index(transition_matrix)%4].set_xlabel("To behavior")
            # axs[file_list.index(file)//4][file_list.index(file)%4].set_title("Transition Matrix")
        plt.title("Transition Matrix")
        plt.tight_layout()
        plt.savefig(save_file)
        plt.show()

def plot_multiple_heatmaps(transition_matrices, indices, labels= None, figsize= (10,10)):
    n_cols = min(3, len(indices))  # Limit to 4 columns max
    n_rows = math.ceil(len(indices) / n_cols)
    
    # Behavior state labels - using shortened versions to avoid overlap
    states = ['Seq.', 'Mov→Tr', 'Drink F', 
              'Mov→Lev','Drink E', 'Off T']
    
    # Find global min and max for consistent color scaling
    matrices_to_plot = [transition_matrices[idx]/3 for idx in indices]
    vmin = 0
    vmax = np.max(matrices_to_plot)
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Create figure and subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Handle case with only one plot
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1 or n_cols == 1:
        axs = axs.reshape(-1, 1) if n_cols == 1 else axs.reshape(1, -1)
    
    # Plot each heatmap
    for i, idx in enumerate(indices):
        row_idx = i // n_cols
        col_idx = i % n_cols
        
        # Get current axis
        ax = axs[row_idx, col_idx]
        
        # Plot heatmap without individual colorbars
        sns.heatmap(matrices_to_plot[i], annot=True, fmt=".1f", cmap='Oranges', 
                   square=True, xticklabels=states, yticklabels=states, 
                   ax=ax, cbar=False, vmin=vmin, vmax=vmax)
        
        # Set title if labels are provided
        if labels is not None and i < len(labels):
            ax.set_title(f"{labels[i]}")
        else:
            ax.set_title(f"Matrix {idx}")
        
        # Only show y-labels for leftmost plots
        if col_idx == 0:
            ax.set_ylabel('From behavior')
        else:
            ax.set_ylabel('')
            
        # Only show x-labels for bottom plots
        if row_idx == n_rows - 1:
            ax.set_xlabel('To behavior')
        else:
            ax.set_xlabel('')
        
        # Rotate tick labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Hide unused subplots if any
    for i in range(len(indices), n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        axs[row_idx, col_idx].axis('off')
    
    # Create a colorbar at the right side of the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
    sm = plt.cm.ScalarMappable(cmap='Oranges', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    
    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar
    plt.savefig(r"behavioral_data\behavior descriptions\full session\stat_graphs\transition\\R_D_heatmap.png")
    plt.show()


    

def chord(matrix, behaviors, exp_name, save_folder = None):
    labels = []
    save_file = os.path.join(save_folder, exp_name + "_chord.png")
    for i, row in enumerate(matrix):
        total = np.sum(row)
        labels.append(f"{behaviors[i]}\n({round(total, 1)})")

    matrix_df = pd.DataFrame(matrix, columns=labels, index=labels)

    colors = {}
    sectors = matrix_df.index.tolist()  # Get your sector names
    color_list = ["green", "darkred", "purple", "navy", "orange", "gray"]

    # Map each sector to a color
    for i, sector in enumerate(sectors):
        colors[sector] = color_list[i % len(color_list)] 

    circos = Circos.initialize_from_matrix(
        matrix_df,
        space=3,
        r_lim=(93,100),
        cmap=colors,
        label_kws=dict(r=105, size= 12, color="black"),
        link_kws=dict(direction=1, ec="black", lw=0.5),
    )

    fig = circos.plotfig()
    plt.title("Behaviors transition, " + exp_name, fontsize=16, fontweight="bold")
    plt.show()
    circos.savefig(save_file)

def markov(matrix, behavior_labels, max_transitions, save_folder = None, exp_num = None):
    save_file = os.path.join(save_folder, exp_num + "_markov.png")
    mc = MarkovChain(matrix, behavior_labels, node_radius=0.6, node_fontsize=10, transparency_func=lambda p: p/max_transitions)
    mc.draw(save_file)
    return mc

def main():
    mice = ["M2", "M4", "M15"]
    exp_list = ["Exp 10", "Exp 11", "Exp 12", "Exp 13", "Exp 14", "Exp 16"]
    full_cycle_order = ['Sequence ', 'Moving To Trough ', 'Drinking Full ', 'Moving To Lever ', 'Drinking Empty ', 'Off Task ']
    transition_matrices = [np.zeros((6,6)) for _ in range(len(exp_list))]

    for mouse in mice:
        file_list = [r"behavioral_data\behavior descriptions\final_description\\" + mouse + os.sep +  f for f in os.listdir(r"behavioral_data\behavior descriptions\final_description\\" + mouse)]
        file_list = file_list[:5] + file_list[6:7]
        for i, file in enumerate(file_list):
            matrix = get_transition_matrix(file, full_cycle_order)
            transition_matrices[i] += matrix

    behavior_labels = ['Sequence ', 'Moving To\nTrough ', 'Drinking Full ', 'Moving To\nLever ','Drinking Empty ', 'Off Task ']
    max_transitions = np.max(transition_matrices)
    for i, matrix in enumerate(transition_matrices) : 
        matrix = matrix/3
        folder = r"behavioral_data\behavior descriptions\behavior_stat_graphs\Transitions"
        chord(matrix, full_cycle_order, exp_list[i], save_folder=folder)
        # markov(matrix, behavior_labels, max_transitions, save_folder= folder, exp_num=exp_list[i])
        # heatmap([matrix], folder, exp_list[i])
    
    # plot_multiple_heatmaps(transition_matrices, [0, 1, 2, 3, 4, 5], labels = ["First FR1 session", "Second FR1 session", "Third FR1 session", "Fourth FR1 session", "Fifth FR1 session", "Sixth FR1 session"])
    

if __name__ == "__main__":
    main()

