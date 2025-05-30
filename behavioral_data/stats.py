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
    intervals_duration = np.abs(end - start)
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
    
    # plt.title(title, fontweight="bold")
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

    exp_list = ["6th FR1", "Complete devaluation"] #[FR1[4]] + random_omission + [FR1[5]] + complete_devaluation

    for i, mouse in enumerate(mice) : 
        behavior_folder = os.path.join(folder_path, mouse)
        behavior_file_list = get_file_list(behavior_folder, ".csv")
        FR1_file_list, random_omission_file_list, complete_devaluation_file_list = get_specific_file_ls(behavior_file_list)
        mouse_times = []
        mouse_task_times = []

        file_list = [FR1_file_list[5]] + complete_devaluation_file_list # [FR1_file_list[1]] + complete_devaluation_file_list
        print(file_list)
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
                start_task_times = behavior_description[task + " Start"].dropna()
                end_task_times = behavior_description[task + " End"].dropna()
                total_task_time = get_total_time(start_task_times, end_task_times, 1000)
                session_task_times.append(total_task_time)

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
    plt.xticks(np.arange(n_sessions), exp_list)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    behavior_handles = handles[:n_behaviors]
    behavior_labels = labels[:n_behaviors]

    
    plt.legend(behavior_handles, behavior_labels, loc='upper right')

    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig(f"behavioral_data\\behavior descriptions\\behavior_stat_graphs\\random_behavior_times_barplot_3_mice.png")


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
    plt.xticks(np.arange(n_sessions), exp_list)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    task_handles = handles[:n_tasks]
    task_labels = labels[:n_tasks]

    
    plt.legend(task_handles, task_labels, loc='upper right')

    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig(f"behavioral_data\\behavior descriptions\\behavior_stat_graphs\\random_task_times_barplot_3_mice.png")
    plt.show()
        
def plot_time_stats_divided():
    behaviors = ["Drinking Full", "Drinking Empty", "Off Task", "Sequence", "Moving To Lever", "Moving To Trough"]
    tasks = ["In Lever Task", "In Trough Task", "Off Task"]
    mice = ["M2", "M4", "M15"]

    folder_path = r"behavioral_data\behavior descriptions\divided_descriptions"
    behavior_times = []
    task_times = []

    exp_list = ["6th FR1", "Complete devaluation", "6th FR1", "Complete devaluation", "6th FR1", "Complete devaluation"]
    title_labels = ["1st subdivision", "2nd subdivision", "3rd subdivision"]

    for i, mouse in enumerate(mice) : 
        behavior_folder = os.path.join(folder_path, mouse)
        behavior_file_list = get_file_list(behavior_folder, ".csv")[6:]
        segment1_files = behavior_file_list[::3]
        segment2_files = behavior_file_list[1::3]
        segment3_files = behavior_file_list[2::3]
        print(segment1_files)

        real_order = segment1_files + segment2_files + segment3_files
        mouse_times = []
        mouse_task_times = []

        for j, file in enumerate(real_order):
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

    mean_behavior_times = np.mean(behavior_times, axis=0)
    mean_task_times = np.mean(task_times, axis=0)

    task_colors = ["red", "blue", "gray"]
    behavior_colors =  ["purple", "orange", "gray", "green", "darkred", "navy"]

    plt.figure(figsize=(15, 8))
    n_sessions = len(exp_list)
    n_behaviors = len(behaviors)
    bar_width = 0.7 / n_behaviors

    for j, behavior in enumerate(behaviors):
        x_pos = np.arange(n_sessions) + j * bar_width - (n_behaviors - 1) * bar_width / 2
        plt.bar(x_pos, mean_behavior_times[:, j], width=bar_width, label=behavior, alpha=0.7, color=behavior_colors[j])

        for m, mouse in enumerate(mice):
            plt.scatter(x_pos, behavior_times[m, :, j], color="black", s=5)
                
    for i in range(1, len(title_labels)):
        line_pos = (i * 2) - 0.5  # Position lines between groups of 2
        plt.axvline(x=line_pos, color='black', linestyle='-', alpha=0.8, linewidth=1.5)

    plt.xlabel("Session")
    plt.ylabel("Time (s)")
    plt.xticks(np.arange(n_sessions), exp_list)

    handles, labels = plt.gca().get_legend_handles_labels()
    behavior_handles = handles[:n_behaviors]
    behavior_labels = labels[:n_behaviors]

    
    plt.legend(behavior_handles, behavior_labels, loc='upper right')

    ax = plt.gca()
    
    # Create secondary x-axis for subdivision labels at the top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    # Position subdivision labels at the center of each group
    subdivision_positions = [0.5, 2.5, 4.5]  # Centers of each pair
    ax2.set_xticks(subdivision_positions)
    ax2.set_xticklabels(title_labels)
    ax2.tick_params(axis='x', which='major', pad=10)  # Add some padding
    
    # Style the subdivision labels
    for tick in ax2.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')
        tick.set_color('black')
    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig(f"behavioral_data\\behavior descriptions\\behavior_stat_graphs\\random_divided_behavior_times_barplot_3_mice.png")
    # plt.show()


    plt.figure(figsize=(15, 8))
    n_tasks = len(tasks)
    bar_width = 0.7 / n_tasks

    for j, task in enumerate(tasks):
        x_pos = np.arange(n_sessions) + j * bar_width - (n_tasks - 1) * bar_width / 2
        plt.bar(x_pos, mean_task_times[:, j], width=bar_width, label=task, alpha=0.7, color=task_colors[j])

        for m, mouse in enumerate(mice):
            plt.scatter(x_pos, task_times[m, :, j], color="black", s=5)
    
    for i in range(1, len(title_labels)):
        line_pos = (i * 2) - 0.5  # Position lines between groups of 2
        plt.axvline(x=line_pos, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
    
    plt.xlabel("Session")
    plt.ylabel("Time (s)")
    plt.xticks(np.arange(n_sessions), exp_list)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    task_handles = handles[:n_tasks]
    task_labels = labels[:n_tasks]

    
    plt.legend(task_handles, task_labels, loc='upper right')

    ax = plt.gca()
    
    # Create secondary x-axis for subdivision labels at the top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    
    # Position subdivision labels at the center of each group
    subdivision_positions = [0.5, 2.5, 4.5]  # Centers of each pair
    ax2.set_xticks(subdivision_positions)
    ax2.set_xticklabels(title_labels)
    ax2.tick_params(axis='x', which='major', pad=10)  # Add some padding
    
    # Style the subdivision labels
    for tick in ax2.get_xticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight('bold')
        tick.set_color('black')

    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig(f"behavioral_data\\behavior descriptions\\behavior_stat_graphs\\random_divided_task_times_barplot_3_mice.png")
    plt.show()
    

def behavior_episodes():
    mice = ["M2", "M4", "M15"]
    exp_list = ["5th FR1", "Random omission"]
    behaviors = ['Sequence', 'Moving To Lever', 'Moving To Trough', 'Drinking Full', 'Drinking Empty', 'Off Task']

    session_durations = np.zeros((len(mice), len(exp_list), len(behaviors)))

    for i, mouse in enumerate(mice):
        folder = f"behavioral_data\\behavior descriptions\\final_description\\{mouse}"
        file_list = get_file_list(folder, ".csv")
        FR1_file_list, random_omission_file_list, complete_devaluation_file_list = get_specific_file_ls(file_list)

        file_list = [FR1_file_list[4]] + random_omission_file_list
        print(file_list)
        for j, f in enumerate(file_list):
            data = pd.read_csv(f)

            for k, behavior in enumerate(behaviors):
                episodes = data[[behavior + " Start", behavior + " End"]].dropna()
                durations = episodes.iloc[:, 1] - episodes.iloc[:, 0]
                mean_duration = np.median(durations)
                if mean_duration > 70000:
                    print(mouse)
                if not durations.empty:
                    mean_duration = np.mean(durations)
                    session_durations[i,j,k] = mean_duration
    
    mean_durations = np.mean(session_durations, axis=0)
    behavior_colors =  ["green", "darkred", "navy", "purple", "orange", "gray"]

    plt.figure(figsize=(15, 8))
    n_sessions = len(exp_list)
    n_behaviors = len(behaviors)
    bar_width = 0.7 / n_behaviors

    for j, behavior in enumerate(behaviors):
        x_pos = np.arange(n_sessions) + j * bar_width - (n_behaviors - 1) * bar_width / 2
        plt.bar(x_pos, mean_durations[:, j], width=bar_width, label=behavior, alpha=0.7, color=behavior_colors[j])

        for m, mouse in enumerate(mice):
            plt.scatter(x_pos, session_durations[m, :, j], color="black", s=5)
                
    
    plt.xlabel("Session")
    plt.ylabel("Time (ms)")
    plt.xticks(np.arange(n_sessions), exp_list)
    plt.ylim(0,45000)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    behavior_handles = handles[:n_behaviors]
    behavior_labels = labels[:n_behaviors]

    
    plt.legend(behavior_handles, behavior_labels, loc='upper right')

    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig(f"behavioral_data\\behavior descriptions\\behavior_stat_graphs\\random_behavior_times_barplot_3_mice.png")
    plt.show()

def behavior_episodes_divided():
    mice = ["M2", "M4", "M15"]
    exp_list = ["5th FR1", "Random omission"]
    behaviors = ['Sequence', 'Moving To Lever', 'Moving To Trough', 'Drinking Full', 'Drinking Empty', 'Off Task']

    folder_path = r"behavioral_data\behavior descriptions\divided_descriptions"
    behavior_times = []

    session_durations = np.zeros((len(mice), len(exp_list), len(behaviors)))

    for i, mouse in enumerate(mice):
        folder = f"behavioral_data\\behavior descriptions\\divided_descriptions\\{mouse}"
        file_list = get_file_list(folder, ".csv")
        FR1_file_list, random_omission_file_list, complete_devaluation_file_list = get_specific_file_ls(file_list)

        file_list = [FR1_file_list[4]] + random_omission_file_list
        print(file_list)
        for j, f in enumerate(file_list):
            data = pd.read_csv(f)

            for k, behavior in enumerate(behaviors):
                episodes = data[[behavior + " Start", behavior + " End"]].dropna()
                durations = episodes.iloc[:, 1] - episodes.iloc[:, 0]
                mean_duration = np.median(durations)
                if mean_duration > 70000:
                    print(mouse)
                if not durations.empty:
                    mean_duration = np.mean(durations)
                    session_durations[i,j,k] = mean_duration
    
    mean_durations = np.mean(session_durations, axis=0)
    behavior_colors =  ["green", "darkred", "navy", "purple", "orange", "gray"]

    plt.figure(figsize=(15, 8))
    n_sessions = len(exp_list)
    n_behaviors = len(behaviors)
    bar_width = 0.7 / n_behaviors

    for j, behavior in enumerate(behaviors):
        x_pos = np.arange(n_sessions) + j * bar_width - (n_behaviors - 1) * bar_width / 2
        plt.bar(x_pos, mean_durations[:, j], width=bar_width, label=behavior, alpha=0.7, color=behavior_colors[j])

        for m, mouse in enumerate(mice):
            plt.scatter(x_pos, session_durations[m, :, j], color="black", s=5)
                
    
    plt.xlabel("Session")
    plt.ylabel("Time (ms)")
    plt.xticks(np.arange(n_sessions), exp_list)
    plt.ylim(0,45000)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    behavior_handles = handles[:n_behaviors]
    behavior_labels = labels[:n_behaviors]

    
    plt.legend(behavior_handles, behavior_labels, loc='upper right')

    
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.savefig(f"behavioral_data\\behavior descriptions\\behavior_stat_graphs\\random_behavior_times_barplot_3_mice.png")
    plt.show()



def plot_press_stats():
    FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]
    random_omission = ["Exp 015"]
    complete_devaluation = ["Exp 017"]
    mice = ["M2", "M4", "M15"]
    exp_list = ["6th FR1", "Complete devaluation"]

    folder_path = r"behavioral_data\behavior descriptions\final_description"
    mean_presses = np.zeros([len(exp_list), len(mice)])
    total_presses = np.zeros([len(exp_list), len(mice)])
    all_seq = [[] for exp in exp_list]
    for i, mouse in enumerate(mice) : 
        behavior_folder = os.path.join(folder_path, mouse)
        behavior_file_list = get_file_list(behavior_folder, ".csv")
        FR1_file_list, random_omission_file_list, complete_devaluation_file_list = get_specific_file_ls(behavior_file_list)

        file_list = [FR1_file_list[5]] + complete_devaluation_file_list #[FR1_file_list[5]] + complete_devaluation_file_list
        
        n_presses = []
        for j, file in enumerate(file_list):
            behavior_description = load_behavior_description(file)
            press_start = behavior_description["Pressing Start"].dropna()
            seq_start = behavior_description["Sequence Start"].dropna()
            seq_end = behavior_description["Sequence End"].dropna()
            press_per_seq = get_presses_per_seq(press_start, seq_start, seq_end)
            n_presses.append(press_per_seq)
            all_seq[j].extend(press_per_seq)
            mean = np.mean(press_per_seq)
            total = np.sum(press_per_seq)
            mean_presses[j,i] = mean
            total_presses[j,i] = total

        
        # plot_boxplot(np.array(n_presses), 
        #              f"Presses per Action Sequence in FR1 sessions ({mouse})", 
        #              "Number of presses", "Session number", 
        #             #  save_path= r"behavioral_data\behavior descriptions\full session\stat_graphs" + f"\\{mouse}_press_per_seq_boxplots.png", 
        #              experiment_list=exp_list
        #              )

    plot_boxplot(mean_presses.T, 
                 "Box plots of presses per sequence (mean of session, all mice)", 
                 "Number of presses", 
                 "Session number", 
                 save_path=r"behavioral_data\behavior descriptions\behavior_stat_graphs" + f"\\devaluation_press_per_seq_boxplots_3_mice.png",
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

    plt.figure(figsize=(10, 6))
    
    # Create the boxplot
    ax = sns.boxplot(data=all_seq)
    # sns.swarmplot(data=all_seq, color='black', size=4, alpha=0.6)
    # Set the x-tick labels if experiment_list is provided
    if exp_list:
        # Make sure we have the right number of labels for the data
        if len(exp_list) == len(all_seq):
            ax.set_xticks(range(len(exp_list)), labels=exp_list)
            # ax.set_xticklabels(experiment_list)
        else:
            # If lengths don't match, use the experiment numbers as labels
            labels = [f"Session {i+1}" for i in range(len(all_seq))]
            ax.set_xticklabels(labels)
            print(f"Warning: Number of experiments ({len(exp_list)}) doesn't match data length ({len(all_seq)})")
    
    # plt.title(title, fontweight="bold")
    plt.ylabel("Number of presses")
    plt.xlabel("Session number")
    ax.set_ylim([0, 5])  
    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()



def plot_press_stats_divided():
    mice = ["M2", "M4", "M15"]
    exp_list = ["6th FR1", "Complete devaluation"]
    n = 3
    folder_path = r"behavioral_data\behavior descriptions\divided_descriptions"
    mean_presses = np.zeros([n, len(exp_list), len(mice)])
    total_presses = [[[] for i in range(n)] for j in range(len(exp_list))]
    print(total_presses)

    for i, mouse in enumerate(mice) : 
        behavior_folder = os.path.join(folder_path, mouse)
        behavior_file_list = get_file_list(behavior_folder, ".csv")
        FR1_file_list, random_omission_file_list, complete_devaluation_file_list = get_specific_file_ls(behavior_file_list)
        
        file_list = FR1_file_list[3:] + complete_devaluation_file_list
        print(file_list)
        for j, file in enumerate(file_list):
            if j < n: k=0
            else: 
                k=1
                j -=n

            behavior_description = load_behavior_description(file)
            press_start = behavior_description["Pressing Start"].dropna()
            seq_start = behavior_description["Sequence Start"].dropna()
            seq_end = behavior_description["Sequence End"].dropna()
            press_per_seq = get_presses_per_seq(press_start, seq_start, seq_end)
            
            mean = np.mean(press_per_seq)
            mean_presses[j,k,i] = mean
            total_presses[k][j].extend(press_per_seq)



    dict_mean= {
        "Session division" : [],
        "Session" : [],
        "Number of presses" : [],
        "Mouse" : []
    }
    for i in range(n):
        for j in range(len(exp_list)):
            values = mean_presses[i,j,:]
            for mouse_idx, value in enumerate(values):
                dict_mean["Session division"].append(i+1)
                dict_mean["Session"].append(exp_list[j])
                dict_mean["Number of presses"].append(value)
                dict_mean["Mouse"].append(mice[mouse_idx])


    dict_all = {
        "Session division" : [],
        "Session" : [],
        "Number of presses" : []
    }
    
    for session_idx, session_name in enumerate(exp_list):  # Loop through sessions
        for division_idx in range(n):  # Loop through session divisions
            values = total_presses[session_idx][division_idx]  # Get all individual presses
            for value in values:  # Add each individual press as a data point
                dict_all["Session division"].append(division_idx + 1)
                dict_all["Session"].append(session_name)
                dict_all["Number of presses"].append(value)

    df_mean = pd.DataFrame(dict_mean)
    df_all = pd.DataFrame(dict_all)
    
    plt.figure(figsize=(10, 6))
    
    # Create the boxplot
    ax = sns.boxplot(data=df_mean, x="Session division", y="Number of presses", hue="Session", palette=["lightcoral", "lightblue"], gap=.1)
    
    mouse_colors = ['C0', 'C1', 'C2']  # Or use sns.color_palette("Set2", 3)
    
    # Manually plot points for each mouse to maintain proper x-positioning
    for mouse_idx, mouse in enumerate(mice):
        mouse_data = df_mean[df_mean["Mouse"] == mouse]
        
        # Plot points for this mouse with consistent color
        for _, row in mouse_data.iterrows():
            session_div = row["Session division"]
            session_type = row["Session"]
            y_val = row["Number of presses"]
            
            # Calculate x position based on session division and session type
            x_base = session_div - 1  # Base x position (0, 1, 2)
            
            # Adjust x position based on session type (dodge like the boxplot)
            if session_type == exp_list[0]:  # "6th FR1"
                x_pos = x_base - 0.2
            else:  # "Complete devaluation"
                x_pos = x_base + 0.2
            
            # Plot the point
            ax.scatter(x_pos, y_val, color=mouse_colors[mouse_idx], 
                      s=64, alpha=0.8, zorder=10, label=mouse if session_div == 1 and session_type == exp_list[0] else "")


    plt.ylabel("Number of presses")
    plt.xlabel("Session division")
    
    # Handle legends
    handles, labels = ax.get_legend_handles_labels()
    # Split legends for Session and Mouse
    session_legend = plt.legend(handles[:2], labels[:2], title="Session", loc='upper left')
    plt.gca().add_artist(session_legend)
    mouse_legend = plt.legend(handles[2:], labels[2:], title="Mouse", loc='upper right')
    
    plt.tight_layout()
    plt.show()



    plt.figure(figsize=(10, 6))
    
    # Create the boxplot
    ax = sns.boxplot(data=dict_all, x="Session division", y="Number of presses", hue="Session", palette=["lightcoral", "lightblue"], gap=.1)

    plt.ylabel("Number of presses")
    plt.xlabel("Session division")
    
    # Handle legends
    handles, labels = ax.get_legend_handles_labels()
    # Split legends for Session and Mouse
    session_legend = plt.legend(handles[:2], labels[:2], title="Session", loc='upper left')
    plt.gca().add_artist(session_legend)
    
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    mouse_colors = ['C0', 'C1', 'C2']
    
    for session_idx, session_name in enumerate(exp_list):
        ax = axes[session_idx]
        
        # Plot line for each mouse
        for mouse_idx, mouse in enumerate(mice):
            # Get data for this mouse and session
            mouse_session_data = df_mean[(df_mean["Mouse"] == mouse) & (df_mean["Session"] == session_name)]
            
            # Sort by session division to ensure proper line connection
            mouse_session_data = mouse_session_data.sort_values("Session division")
            
            # Plot the line
            ax.plot(mouse_session_data["Session division"], 
                   mouse_session_data["Number of presses"], 
                   marker='o', 
                   color=mouse_colors[mouse_idx], 
                   linewidth=2, 
                   markersize=8, 
                   label=mouse,
                   alpha=0.8)
        
        ax.set_xlabel("Session division")
        ax.set_ylabel("Mean number of presses")
        ax.set_title(f"{session_name}")
        ax.legend(title="Mouse")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, n+1))
    
    plt.tight_layout()
    plt.show()


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
    
    # plt.savefig(r"behavioral_data\behavior descriptions\behavior_stat_graphs\FR1_n_full_task_cycles.png")
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
    median = np.median(time_per_seq)

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
    plt.vlines(median, 0, np.max(counts), colors='black', linestyles="dashed", label=f"Median value = {round(median, 2)} ms")
    plt.xlabel('Time duration of complete task cycles (ms)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    # plt.savefig(r"behavioral_data\behavior descriptions\behavior_stat_graphs\dz_time_full_seq.png")
    plt.show()

def plot_time_behaving_optimally():
    FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]
    random_omission = ["Exp 015"]
    complete_devaluation = ["Exp 017"]
    mice = ["M2", "M4", "M15"]
    median_time = [28603.0, 19142.0, 11555.0]

    exp_list = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "6th FR1"] #[FR1[4]] + random_omission + [FR1[5]] + complete_devaluation

    duration = pd.read_csv(r"behavioral_data\behavior descriptions\session_duration.csv")
    selected_rows = duration.iloc[[0, 1, 2, 3, 4, 6]]
    selected_array = selected_rows.to_numpy()

    cycles_per_mouse = np.zeros([len(exp_list), len(mice)])
    perc_in_task = np.zeros([len(exp_list), len(mice)])

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
            perc_in_task[j, i] = n_cycles*median_time[i]/selected_array[j,i]*100
    
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
    # plt.savefig(f"behavioral_data\\behavior descriptions\\behavior_stat_graphs\\optimal_behaving_times_barplot_3_mice.png")
    plt.show()

def plot_visists_and_lever():
    
    mice = ["M2", "M4", "M15"]
    exp_list = ["fifth FR1", "Random omission"]
    stats = ["Number of Sequences", "Dispenser visits"]

    all_seq = np.zeros((len(mice), len(exp_list)))
    all_visits = np.zeros((len(mice), len(exp_list)))
    empty_visits = np.zeros((len(mice), len(exp_list)))
    full_visits = np.zeros((len(mice), len(exp_list)))

    for i, mouse in enumerate(mice):
        folder = f"behavioral_data\\behavior descriptions\\final_description\\{mouse}"
        file_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')][5:6]
        print(file_list)

        for j, f in enumerate(file_list):
            data = pd.read_csv(f)
            n_seq = len(data["Sequence Start"].dropna())
            n_visits = len(data["Drinking Full Start"].dropna()) + len(data["Drinking Empty Start"].dropna())
            n_full = len(data["Drinking Full Start"].dropna())
            n_empty = len(data["Drinking Empty Start"].dropna())
            all_seq[i,j] = n_seq
            all_visits[i, j] = n_visits
            full_visits[i,j] = n_full
            empty_visits[i,j] = n_empty

    print(full_visits)
    print(empty_visits)

    mean_seq = np.mean(all_seq, axis=0)
    mean_visits = np.mean(all_visits, axis=0)

    # Combine the data for plotting
    all_data = [all_seq, all_visits]
    mean_data = [mean_seq, mean_visits]

    plt.figure(figsize=(10, 5))
    n_sessions = len(exp_list)
    n_stats = len(stats)
    bar_width = 0.35  # Adjust bar width for better spacing
    colors = ["green", "purple"]

    for j, stat in enumerate(stats):
        x_pos = np.arange(n_sessions) + j * bar_width - (n_stats - 1) * bar_width / 2
        plt.bar(x_pos, mean_data[j], width=bar_width, label=stat, alpha=0.7, color=colors[j])

        # Plot individual mouse data points
        for m, mouse in enumerate(mice):
            plt.scatter(x_pos, all_data[j][m], color="black", s=30, alpha=0.8)

    plt.xlabel("Session")
    plt.ylabel("Count")  # Changed from "Time (s)" to "Count" since these are counts
    plt.xticks(np.arange(n_sessions), exp_list)

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_lick_duration():
    mice = ["M2", "M4", "M15"]

    all_durations = []

    all_full_durations = []
    all_empty_durations = []
    for mouse in mice:
        folder = f"behavioral_data\\behavior descriptions\\final_description\\{mouse}"
        file_list = get_file_list(folder, '.csv')

        for f in file_list:
            data = pd.read_csv(f)
            full_duration = data["Licking Full End"].dropna() - data["Licking Full Start"].dropna()
            empty_duration = data["Licking Empty End"].dropna() - data["Licking Empty Start"].dropna()
            
            all_full_durations.extend(full_duration/1000)
            all_empty_durations.extend(empty_duration/1000)

            all_durations.extend(full_duration/1000)
            all_durations.extend(empty_duration/1000)

    all_durations = np.array(all_durations)
    print(np.max(all_durations), np.min(all_durations), np.mean(all_durations))
    below_01 = len(all_durations[all_durations<=0.2])/len(all_durations)
    print(below_01)
    plt.figure()
    plt.hist(all_durations, bins=500,  alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel("Lick duration (s)")
    plt.title(f"Licking Duration Distribution\n(n={len(all_durations)} licks)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    mean_full = np.mean(all_durations)
    median_full = np.median(all_durations)
    plt.axvline(mean_full, color='red', linestyle='--', label=f'Mean: {mean_full:.3f}s')
    plt.axvline(median_full, color='orange', linestyle='--', label=f'Median: {median_full:.3f}s')
    plt.legend()
    
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot full licking durations
    if all_full_durations:
        ax1.hist(all_full_durations, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title(f'Full Licking Duration Distribution\n(n={len(all_full_durations)} licks)')
        ax1.set_xlabel('Duration (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_full = np.mean(all_full_durations)
        median_full = np.median(all_full_durations)
        ax1.axvline(mean_full, color='red', linestyle='--', label=f'Mean: {mean_full:.3f}s')
        ax1.axvline(median_full, color='orange', linestyle='--', label=f'Median: {median_full:.3f}s')
        ax1.legend()
    
    # Plot empty licking durations
    if all_empty_durations:
        ax2.hist(all_empty_durations, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax2.set_title(f'Empty Licking Duration Distribution\n(n={len(all_empty_durations)} licks)')
        ax2.set_xlabel('Duration (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_empty = np.mean(all_empty_durations)
        median_empty = np.median(all_empty_durations)
        ax2.axvline(mean_empty, color='red', linestyle='--', label=f'Mean: {mean_empty:.3f}s')
        ax2.axvline(median_empty, color='orange', linestyle='--', label=f'Median: {median_empty:.3f}s')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()


plot_visists_and_lever()


