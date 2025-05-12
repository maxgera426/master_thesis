import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns



def load_behavior_description(file_path):
    return pd.read_csv(file_path)

def get_file_list(folder_path, file_extension):
    return [os.path.join(folder_path, file) 
                for file in os.listdir(folder_path) 
                if os.path.splitext(file)[1] == file_extension
            ]

def get_transition_df(file_path):
    data = pd.read_csv(file_path)
    states = ["Sequence", "Moving To Trough", "Drinking Full", "Moving To Lever", "Drinking Empty", "Off Task"]
    

    state_df = pd.DataFrame(columns=["state", "start", "end"])
    for state in states:
        start_times = data[state + " Start"].dropna()
        end_times = data[state + " End"].dropna()
        for start, end in zip(start_times, end_times):
            state_df = state_df._append({"state": state, "start": start, "end" : end}, ignore_index=True)
    state_df = state_df.sort_values(by="start").reset_index(drop=True)

    return state_df

def get_specific_file_ls(file_list):
    FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]
    random_omission = ["Exp 015"]
    complete_devaluation = ["Exp 017"]

    FR1_file_list = []
    random_omission_file_list = []
    complete_devaluation_file_list = []
    for file_path in file_list:
        if any(exp in file_path for exp in FR1):
            FR1_file_list.append(file_path)
        elif any(exp in file_path for exp in random_omission):
            random_omission_file_list.append(file_path)
        elif any(exp in file_path for exp in complete_devaluation):
            complete_devaluation_file_list.append(file_path)
    return FR1_file_list, random_omission_file_list, complete_devaluation_file_list



def get_total_time(start, end, units=None):
    intervals_duration = end - start
    total_time = np.sum(intervals_duration)
    if units:
        total_time /= units
    return total_time

def get_presses_per_seq(presses_start, seq_start, seq_end):
    list_n_presses = []
    if not seq_start.empty:
        seq_end = seq_end[seq_end>= seq_start.iloc[0]]
    else:
        print("No action sequence in this time interval")
        list_n_presses.append(0)
    for start, end in zip(seq_start, seq_end):
        presses = presses_start[(presses_start >= start) & (presses_start <= end)]
        n_presses = len(presses)
        list_n_presses.append(n_presses)

    return list_n_presses

def seq_counter(state_data, sequence):
    big_counter = 0
    counter = 0

    for _, row in state_data.iterrows():
        if row["state"] in sequence[counter]:
            counter += 1
        else :
            counter = 0
        if counter == len(sequence):
            counter = 0
            big_counter += 1

    return big_counter

def time_of_seq(state_data, sequence):
    counter = 0
    time_list = []

    for _, row in state_data.iterrows():
        if row["state"] == sequence[0]:
            start = row["start"]
        if row["state"] == sequence[-1]:
            end = row["end"]

        if row["state"] in sequence[counter]:
            counter += 1
        else :
            counter = 0
        if counter == len(sequence):
            duration = end - start
            time_list.append(duration)
            counter = 0
    
    return time_list


def plot_boxplot(data, title, y_label, x_label, save_path=None, experiment_list=None):
    plt.figure(figsize=(10, 6))
    
    # Create the boxplot
    ax = sns.boxplot(data=data)
    sns.swarmplot(data=data, color='black', size=4, alpha=0.6)
    # Set the x-tick labels if experiment_list is provided
    if experiment_list:
        # Make sure we have the right number of labels for the data
        if len(experiment_list) == len(data.T):
            ax.set_xticks(range(len(experiment_list)), labels=experiment_list)
            # ax.set_xticklabels(experiment_list)
        else:
            # If lengths don't match, use the experiment numbers as labels
            labels = [f"Session {i+1}" for i in range(len(data))]
            ax.set_xticklabels(labels)
            print(f"Warning: Number of experiments ({len(experiment_list)}) doesn't match data length ({len(data)})")
    
    plt.title(title, fontweight="bold")
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    ax.set_ylim([0, 5])  
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_time_stats():
    behaviors = ["Drinking Full", "Drinking Empty", "Off Task", "Sequence", "Moving To Lever", "Moving To Trough"]
    tasks = ["In Lever Task", "In Trough Task", "Off Task"]
    mice = ["M2", "M4", "M15"]
    FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]
    random_omission = ["Exp 015"]
    complete_devaluation = ["Exp 017"]
    folder_path = r"behavioral_data\behavior descriptions\final_description"
    behavior_times = []
    task_times = []

    exp_list = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "6th FR1"] #[FR1[4]] + random_omission + [FR1[5]] + complete_devaluation

    for i, mouse in enumerate(mice) : 
        behavior_folder = os.path.join(folder_path, mouse)
        behavior_file_list = get_file_list(behavior_folder, ".csv")
        FR1_file_list, random_omission_file_list, complete_devaluation_file_list = get_specific_file_ls(behavior_file_list)
        mouse_times = []
        mouse_task_times = []

        file_list = FR1_file_list #[FR1_file_list[0]] + random_omission_file_list + [FR1_file_list[1]] + complete_devaluation_file_list

        for j, file in enumerate(file_list):
            behavior_description = load_behavior_description(file)
            session_times = []

            session_task_times = []

            for behavior in behaviors:
                start_times = behavior_description[behavior + " Start"].dropna()
                end_times = behavior_description[behavior + " End"].dropna()
                total_time = get_total_time(start_times, end_times, 1000)
                session_times.append(total_time)
            
            for task in tasks:
                start_times = behavior_description[task + " Start"].dropna()
                end_times = behavior_description[task + " End"].dropna()
                total_time = get_total_time(start_times, end_times, 1000)
                session_task_times.append(total_time)

            mouse_times.append(session_times)
            mouse_task_times.append(session_task_times)

        behavior_times.append(mouse_times)
        task_times.append(mouse_task_times)

    behavior_times = np.array(behavior_times)
    task_times = np.array(task_times)
    mean_times = np.mean(behavior_times, axis=0)
    mean_task_times = np.mean(task_times, axis=0)

    # for i, mouse in enumerate(mice):
    #     plt.figure(figsize=(15, 8))
        
    #     mouse_data = behavior_times[i]

    #     n_sessions = len(exp_list)
    #     n_behaviors = len(behaviors)
    #     bar_width = 0.7 / n_behaviors

    #     for j, behavior in enumerate(behaviors):
    #         x_pos = np.arange(n_sessions) + j * bar_width - (n_behaviors - 1) * bar_width / 2
    #         plt.bar(x_pos, mouse_data[:, j], width=bar_width, label=behavior)
        
    #     plt.title(f"Behavior Times for {mouse}")
    #     plt.xlabel("Session")
    #     plt.ylabel("Time (s)")
    #     plt.xticks(np.arange(n_sessions), exp_list, rotation=45)
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.tight_layout()
    #     plt.grid(axis='y', linestyle='--', alpha=0.7)
    #     # plt.savefig(f"behavioral_data\\behavior descriptions\\full session\\stat_graphs\\{mouse}_behavior_times_barplot.png")
    #     plt.show()

    task_colors = ["red", "blue", "gray"]
    behavior_colors =  ["purple", "orange", "gray", "green", "darkred", "navy"]

    plt.figure(figsize=(15, 8))
    n_sessions = len(exp_list)
    n_behaviors = len(behaviors)
    bar_width = 0.7 / n_behaviors

    for j, behavior in enumerate(behaviors):
        x_pos = np.arange(n_sessions) + j * bar_width - (n_behaviors - 1) * bar_width / 2
        plt.bar(x_pos, mean_times[:, j], width=bar_width, label=behavior, alpha=0.7, color=behavior_colors[j])

        for m, mouse in enumerate(mice):
            plt.scatter(x_pos, behavior_times[m, :, j], color="black", s=5)
                
    
    plt.xlabel("Session")
    plt.ylabel("Time (s)")
    plt.xticks(np.arange(n_sessions), exp_list, rotation=45)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    behavior_handles = handles[:n_behaviors]
    behavior_labels = labels[:n_behaviors]

    
    plt.legend(behavior_handles, behavior_labels, loc='upper right')

    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"behavioral_data\\behavior descriptions\\behavior_stat_graphs\\behavior_times_barplot_3_mice.png")


    plt.figure(figsize=(15, 8))
    n_tasks = len(tasks)
    bar_width = 0.7 / n_tasks

    for j, task in enumerate(tasks):
        x_pos = np.arange(n_sessions) + j * bar_width - (n_tasks - 1) * bar_width / 2
        plt.bar(x_pos, mean_task_times[:, j], width=bar_width, label=task, alpha=0.7, color=task_colors[j])

        for m, mouse in enumerate(mice):
            plt.scatter(x_pos, task_times[m, :, j], color="black", s=5)
                
    
    plt.xlabel("Session")
    plt.ylabel("Time (s)")
    plt.xticks(np.arange(n_sessions), exp_list, rotation=45)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    task_handles = handles[:n_tasks]
    task_labels = labels[:n_tasks]

    
    plt.legend(task_handles, task_labels, loc='upper right')

    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"behavioral_data\\behavior descriptions\\behavior_stat_graphs\\task_times_barplot_3_mice.png")
    plt.show()
        


def plot_press_stats():
    FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]
    random_omission = ["Exp 015"]
    complete_devaluation = ["Exp 017"]
    mice = ["M2", "M4", "M15"]
    exp_list = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "6th FR1"] #[FR1[4]] + random_omission + [FR1[5]] + complete_devaluation

    folder_path = r"behavioral_data\behavior descriptions\final_description"
    mean_presses = np.zeros([len(exp_list), len(mice)])
    total_presses = np.zeros([len(exp_list), len(mice)])

    for i, mouse in enumerate(mice) : 
        behavior_folder = os.path.join(folder_path, mouse)
        behavior_file_list = get_file_list(behavior_folder, ".csv")
        FR1_file_list, random_omission_file_list, complete_devaluation_file_list = get_specific_file_ls(behavior_file_list)

        file_list = FR1_file_list #[FR1_file_list[4]] + random_omission_file_list + [FR1_file_list[5]] + complete_devaluation_file_list
        n_presses = []
        for j, file in enumerate(file_list):
            behavior_description = load_behavior_description(file)
            press_start = behavior_description["Pressing Start"].dropna()
            seq_start = behavior_description["Sequence Start"].dropna()
            seq_end = behavior_description["Sequence End"].dropna()
            press_per_seq = get_presses_per_seq(press_start, seq_start, seq_end)
            n_presses.append(press_per_seq)
            mean = np.mean(press_per_seq)
            total = np.sum(press_per_seq)
            mean_presses[j,i] = mean
            total_presses[j,i] = total

        
        # plot_boxplot(n_presses, 
        #              f"Presses per Action Sequence in FR1 sessions ({mouse})", 
        #              "Number of presses", "Session number", 
        #             #  save_path= r"behavioral_data\behavior descriptions\full session\stat_graphs" + f"\\{mouse}_press_per_seq_boxplots.png", 
        #              experiment_list=exp_list
        #              )

    plot_boxplot(mean_presses.T, 
                 "Box plots of presses per sequence (mean of session, all mice)", 
                 "Number of presses", 
                 "Session number", 
                 save_path=r"behavioral_data\behavior descriptions\full session\stat_graphs\FR1" + f"\\press_per_seq_boxplots_3_mice.png",
                 experiment_list=exp_list
                 )
    
    x = np.arange(len(exp_list)) 
    plt.figure(figsize=(10, 6))
    for i, mouse_data in enumerate(mean_presses.T):
        plt.plot(x, mouse_data, label=mice[i], marker='o', linewidth=2)
    plt.xticks(x, exp_list)
    plt.xlabel("Session number")
    plt.ylabel("Number of presses")
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # x = np.arange(len(exp_list))
    # width = 0.25  # Width of the bars
    # plt.figure(figsize=(12, 6))
    # ax = plt.gca()

    # # Calculate positions for each group of bars
    # for i, mouse in enumerate(total_presses.T):
    #     offset = (i - (len(mice) - 1) / 2) * width
    #     plt.bar(x + offset, mouse, width=width, label=f"{mice[i]}")

    # # Set the x-tick positions and labels
    # ax.set_xticks(x)
    # ax.set_xticklabels(exp_list)

    # plt.xlabel("Session no.")
    # plt.ylabel("Number of presses")
    # plt.title("Total presses per session", fontweight="bold")
    # plt.legend()
    # plt.tight_layout()  # Adjust layout to prevent clipping
    # plt.savefig(r"behavioral_data\behavior descriptions\full session\stat_graphs\random and complete" + f"\\total_presses_barplot.png")
    # plt.show()

def plot_full_seq():
    mice = ["M2", "M4", "M15"]
    exp_list = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "6th FR1"]

    cycles_per_mouse = np.zeros([len(exp_list), len(mice)])

    folder_path = r"behavioral_data\behavior descriptions\final_description"
    full_cycle = ['Sequence ', 'Moving To Trough ', 'Drinking Full ', 'Moving To Lever ']

    for i, mouse in enumerate(mice) : 
        behavior_folder = os.path.join(folder_path, mouse)
        behavior_file_list = get_file_list(behavior_folder, ".csv")
        FR1_file_list, random_omission_file_list, complete_devaluation_file_list = get_specific_file_ls(behavior_file_list)

        file_list = FR1_file_list #[FR1_file_list[0]] + random_omission_file_list + [FR1_file_list[1]] + complete_devaluation_file_list

        for j, file in enumerate(file_list):
            state_df = get_transition_df(file)
            n_cycles = seq_counter(state_df, full_cycle)
            cycles_per_mouse[j, i] = n_cycles

    
    print(cycles_per_mouse, np.sum(cycles_per_mouse))
    means = np.mean(cycles_per_mouse, axis=1)
    x_values = np.arange(len(cycles_per_mouse))

   
    fig, ax = plt.subplots()
    for i, row in enumerate(cycles_per_mouse):
        for j, value in enumerate(row):
            ax.scatter(i+1, row[j], color='white', edgecolors='black', label = [mouse[j]] if i == 0 else "", alpha = 0.8)

    ax.plot(x_values +1, means, color='black', marker='o', linestyle='-')


    ax.set_xlabel('Session no.')
    ax.set_ylabel('No. of complete cycles')

    ax.set_xticks(np.arange(len(exp_list))+1)
    ax.set_xticklabels(exp_list, rotation=45)
    plt.tight_layout()
    
    plt.savefig(r"behavioral_data\behavior descriptions\behavior_stat_graphs\n_full_task_cycles_drinking.png")
    plt.show()

def plot_time_full_seq():
    FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]
    random_omission = ["Exp 015"]
    complete_devaluation = ["Exp 017"]
    mice = ["M2", "M4", "M15"]

    folder_path = r"behavioral_data\behavior descriptions\final_description"
    full_cycle = ['Sequence', 'Moving To Trough', 'Drinking Full', 'Moving To Lever']
    
    time_per_seq = []

    for mouse in mice:
        behavior_folder = os.path.join(folder_path, mouse)
        behavior_file_list = get_file_list(behavior_folder, ".csv")
        FR1_file_list, random_omission_file_list, complete_devaluation_file_list = get_specific_file_ls(behavior_file_list)

        file_list = FR1_file_list #[FR1_file_list[0]] + random_omission_file_list + [FR1_file_list[1]] + complete_devaluation_file_list
        for file in file_list:
            state_df = get_transition_df(file)
            time_per_seq += time_of_seq(state_df, full_cycle)

    min_val = np.min(time_per_seq)
    max_val = np.max(time_per_seq)
    mean = np.mean(time_per_seq)

    bin_size = 500 
    nbins = (max_val - min_val)//bin_size + 1
    x = np.arange(min_val, max_val, bin_size)
    counts = np.zeros(int(nbins))
    for value in time_per_seq:
        diff = np.abs(x - value)
        i = np.argmin(diff)
        counts[i] += 1

    counts = counts/np.sum(counts)

    plt.figure()
    plt.bar(x, counts, width=bin_size, align='edge', edgecolor='black', color='green')
    plt.vlines(mean, 0, np.max(counts), colors='black', linestyles="dashed", label=f"Mean value = {round(mean, 2)} ms")
    plt.xlabel('Time duration of complete task cycles (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    # plt.savefig(r"behavioral_data\behavior descriptions\behavior_stat_graphs\hist_time_full_seq.png")
    plt.show()

def plot_time_behaving_optimally():
    FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]
    random_omission = ["Exp 015"]
    complete_devaluation = ["Exp 017"]
    mice = ["M2", "M4", "M15"]
    mean_time = 18016.07

    exp_list = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "6th FR1"] #[FR1[4]] + random_omission + [FR1[5]] + complete_devaluation

    cycles_per_mouse = np.zeros([len(exp_list), len(mice)])

    folder_path = r"behavioral_data\behavior descriptions\final_description"
    full_cycle = ['Sequence', 'Moving To Trough', 'Drinking Full', 'Moving To Lever']
    for i, mouse in enumerate(mice) : 
        behavior_folder = os.path.join(folder_path, mouse)
        behavior_file_list = get_file_list(behavior_folder, ".csv")
        FR1_file_list, random_omission_file_list, complete_devaluation_file_list = get_specific_file_ls(behavior_file_list)

        file_list = FR1_file_list #[FR1_file_list[0]] + random_omission_file_list + [FR1_file_list[1]] + complete_devaluation_file_list

        for j, file in enumerate(file_list):
            state_df = get_transition_df(file)
            n_cycles = seq_counter(state_df, full_cycle)
            cycles_per_mouse[j, i] = n_cycles

    duration = pd.read_csv(r"behavioral_data\behavior descriptions\session_duration.csv")
    selected_rows = duration.iloc[[0, 1, 2, 3, 4, 6]]
    selected_array = selected_rows.to_numpy()


    perc_in_task = cycles_per_mouse*mean_time/selected_array*100
    print(cycles_per_mouse)
    print(selected_array)
    mean_percentages = np.mean(perc_in_task, axis=1)
    print(mean_percentages)

    plt.figure(figsize=(15, 8))
    n_sessions = len(exp_list)
    bar_width = 0.5

    x_pos = np.arange(n_sessions)
    plt.bar(x_pos, mean_percentages, width=bar_width, alpha=0.7)

    for m, mouse in enumerate(mice):
        plt.scatter(x_pos, perc_in_task[:, m], color="black", s=5)
                
    
    # plt.title("Optimal behaving time in percentage regarding full session time")
    plt.xlabel("Session")
    plt.ylabel("Percentage (%)")
    plt.xticks(np.arange(n_sessions), exp_list, rotation=45)
    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"behavioral_data\\behavior descriptions\\behavior_stat_graphs\\optimal_behaving_times_barplot_3_mice.png")
    plt.show()


plot_time_behaving_optimally()


