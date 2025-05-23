import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import os

def load_percentile_data(file_path):
    data = pd.read_csv(file_path)
    return data

def get_file_list(folder):
    return [folder + f for f in os.listdir(folder)]

def activity_percentage(significance, categories):
    num_neurons = len(significance)
    # Count separately for positive, negative, and total active neurons
    positive_neurons = np.sum(significance == 1, axis=0)
    negative_neurons = np.sum(significance == -1, axis=0)
    total_active = positive_neurons + negative_neurons
    
    # Calculate percentages
    positive_percent = positive_neurons / num_neurons * 100
    negative_percent = negative_neurons / num_neurons * 100
    total_active_percent = total_active / num_neurons * 100
    
    return positive_percent, negative_percent, total_active_percent

def encode_percentiles(percentile_array, high=95, low=5):
    return np.where(percentile_array > high, 1, np.where(percentile_array < low, -1, 0))

def extract_experiment_name(file_path):
    return os.path.basename(file_path)[:7]

def plot_number_neurons():
    mice = ["M2", "M4", "M15"]
    labels = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "Random omission", "6th FR1", "Complete devaluation"]
    
    neurons = np.zeros((len(mice), len(labels)))
    for i, mouse in enumerate(mice):
        folder = f"neuronal_activity_data\\cell_props\\{mouse}"
        file_list = [os.path.join(folder, f) for f in os.listdir(folder)]
        for j, f in enumerate(file_list):
            data = pd.read_csv(f)
            n_cells = len(data["Name"])
            neurons[i, j] = n_cells

    plt.figure(figsize=(12, 6))

    for i in range(len(mice)):
        plt.plot(range(len(labels)), neurons[i], '-o', label=mice[i])
    
    # Customize the plot
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.ylabel('Number of Neurons')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return

def active_inactive(type):
    mice = ["M2", "M4", "M15"]
    labels = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "6th FR1"]
    FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]

    freq_type = {
        "density": "density_per_behavior",
        "mean": "mean_per_behavior",
        "peak": "peaks_per_behavior"
    }

    n_modulated = np.zeros((len(mice), len(FR1)))
    n_nothing = np.zeros((len(mice), len(FR1)))

    total_n = np.zeros((len(mice), len(FR1)))

    for i, mouse in enumerate(mice):
        folder = f"neuronal_activity_data\\percentiles\\{mouse}\\behavior_related\\{freq_type[type]}"
        FR1_file_list = [os.path.join(folder, f) for f in os.listdir(folder) if any(exp in f for exp in FR1)]

        for j, f in enumerate(FR1_file_list):
            data = pd.read_csv(f)
            n = len(data)
            total_n[i,j] = n
            if not data.empty:
                for _, neuron in data.iterrows():
                    if any(np.abs(val) == 1 for val in neuron.iloc[1:]):
                        n_modulated[i, j] += 1
                    else:
                        n_nothing[i, j] +=1
                    
                    # for v in neuron.iloc[1:] : 
                    #     if v == 1:
                    #         n_active[i,j]+= 1
                    #     elif v == -1:
                    #         n_inactive[i,j] += 1
                    #     elif v == 0:
                    #         n_nothing[i,j] += 1


    print(n_modulated)
    print(n_nothing)

    modulated_neurons = np.sum(n_modulated, axis=0)
    nothing_neurons = np.sum(n_nothing, axis=0)
    total_neurons = 1# np.sum(total_n, axis=0)

    plt.figure()
    x = np.arange(len(FR1))
    plt.plot(x, modulated_neurons/total_neurons, "b-", label="Modulated neurons")
    plt.plot(x, nothing_neurons/total_neurons, "k--", alpha = 0.4, label="Non significant neurons")
    plt.xticks(x, labels)
    plt.legend()
    plt.show()

def activity_behaviors(type):
    behavior_list = ["Sequence", "Moving To Trough", "Locomotion To Trough", "Drinking Full", "Moving To Lever", "Locomotion To Lever", "Drinking Empty", "Off Task"]
    mice = ["M2", "M4", "M15"]
    labels = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "6th FR1"]
    FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]

    freq_type = {
        "density": "density_per_behavior",
        "mean": "mean_per_behavior",
        "peak": "peaks_per_behavior"
    }
    
    active_counts = np.zeros((len(mice), len(FR1), len(behavior_list)))
    inactive_counts = np.zeros((len(mice), len(FR1), len(behavior_list)))
    nothing_counts = np.zeros((len(mice), len(FR1), len(behavior_list)))

    for i, mouse in enumerate(mice):
        folder = f"neuronal_activity_data\\percentiles\\{mouse}\\behavior_related\\{freq_type[type]}"
        FR1_file_list = [os.path.join(folder, f) for f in os.listdir(folder) if any(exp in f for exp in FR1)]

        for j, f in enumerate(FR1_file_list):
            data = pd.read_csv(f)
            
            behaviors = data.columns[1:]

            for k, column in enumerate(behaviors):
                value_counts = data[column].value_counts()
                count_1 = value_counts.get(1, 0)
                count_minus1 = value_counts.get(-1, 0)
                count_0 = value_counts.get(0, 0)

                active_counts[i, j, k] = count_1
                inactive_counts[i, j, k] = count_minus1
                nothing_counts[i, j, k] = count_0

    
    all_mice_active = np.sum(active_counts, axis=0)
    all_mice_inactive = np.sum(inactive_counts, axis=0)
    all_mice_nothing = np.sum(nothing_counts, axis=0)

    fig, ax = plt.subplots(figsize=(15, 8))

    n_behaviors = all_mice_active.shape[1]  # Number of behaviors (columns)
    n_sessions = all_mice_active.shape[0]   # Number of sessions (rows)
    
    bar_width = 0.12
    positions = np.arange(n_behaviors)  # Position for each behavior
    
    # Color schemes for active and inactive
    active_colors = plt.cm.Greens(np.linspace(0.3, 1, n_sessions))
    inactive_colors = plt.cm.Reds(np.linspace(0.3, 1, n_sessions))
    nothing_colors = plt.cm.Grays(np.linspace(0.3, 1, n_sessions))
    
    for i in range(n_sessions):
        offset = -(i - n_sessions/2 + 0.5) * bar_width
        
        # Create horizontal stacked bars
        bars1 = ax.barh(positions + offset, all_mice_active[i], bar_width, 
                       label=f'{labels[i]} - Active' if i == 0 else "", 
                       color=active_colors[i], alpha=0.8)
    
        bars2 = ax.barh(positions + offset, all_mice_inactive[i], bar_width, 
                       left=all_mice_active[i], 
                       label=f'{labels[i]} - Inactive' if i == 0 else "", 
                       color=inactive_colors[i], alpha=0.8)
        
        bars3 = ax.barh(positions + offset, all_mice_nothing[i], bar_width, 
                       left=all_mice_inactive[i] + all_mice_active[i], 
                       label=f'{labels[i]} - Non significant' if i == 0 else "", 
                       color=nothing_colors[i], alpha=0.8)
        
        # Add value labels
        for j, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
            width1 = bar1.get_width()
            width2 = bar2.get_width()
            width3 = bar3.get_width()
            
            # Label for left part (active)
            if width1 > 0:
                ax.text(width1/2, bar1.get_y() + bar1.get_height()/2.,
                       f'{int(width1)}', ha='center', va='center', fontsize=7, fontweight='bold')
            
            # Label for right part (inactive)
            if width2 > 0:
                ax.text(width1 + width2/2, bar2.get_y() + bar2.get_height()/2.,
                       f'{int(width2)}', ha='center', va='center', fontsize=7, fontweight='bold')
            
            # Label for right part (inactive)
            if width3 > 0:
                ax.text(width1 + width2 + width3/2, bar3.get_y() + bar3.get_height()/2.,
                       f'{int(width3)}', ha='center', va='center', fontsize=7, fontweight='bold')
            
            # Total label at the end
            total = width1 + width2 + width3
            if total > 0:
                ax.text(total + 0.5, bar1.get_y() + bar1.get_height()/2.,
                       f'{int(total)}', ha='left', va='center', fontsize=8, style='italic')
    
    # Fixed axis labels and settings for horizontal bars
    ax.set_xlabel('Count', fontsize=12, fontweight='bold')  # Swapped
    ax.set_ylabel('Behaviors', fontsize=12, fontweight='bold')  # Swapped

    # Set y-axis (behaviors) instead of x-axis
    ax.set_yticks(positions)
    ax.set_yticklabels(behavior_list)  # No rotation needed for horizontal
    
    # Create proper legend
    legend_elements = []
    # Add session labels
    for i in range(n_sessions):
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='gray', alpha=0.8, label=labels[i]))
    
    # Add data type labels
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.8, label='Active'))
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.8, hatch='///', label='Inactive'))
    
    ax.legend(handles=legend_elements, title='Sessions & Activity Types', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Grid for horizontal bars should be on x-axis
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def correlation():
    mice = ["M2", "M4", "M15"]
    labels = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "6th FR1"]
    FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]

    mean_correlation = np.zeros((len(mice), len(FR1)))
    correlation_coeffs = [[] for exp in FR1]
    
    for i, mouse in enumerate(mice):
        folder = f"neuronal_activity_data\\calcium_traces\\{mouse}\\status_based_2"
        FR1_file_list = [os.path.join(folder, f) for f in os.listdir(folder) if any(exp in f for exp in FR1)]
        for j, f in enumerate(FR1_file_list):
            traces = pd.read_csv(f)

            if not traces.empty:
                neurons = traces.columns[1:]

                pearson_corr = traces[neurons].corr('pearson')
                top_triangle = np.triu(pearson_corr, 1)
                mean_abs = np.mean(np.abs(top_triangle), where=((top_triangle != 0)))

                if np.isnan(mean_abs):
                    mean_abs = 0
                mean_correlation[i, j] = mean_abs
                
                for r in range(len(neurons)-1):
                    for c in range(r+1, len(neurons)):
                        trace1 = traces[neurons[r]]
                        trace2 = traces[neurons[c]]
                        coeff = np.corrcoef(trace1, trace2)[0, 1]
                        # coeff, _ = stats.spearmanr(trace1, trace2)

                        correlation_coeffs[j].append(coeff)

    all_mice_corr = np.mean(mean_correlation, where=(mean_correlation != 0), axis=0)

    all = 0
    count = 0
    for i in correlation_coeffs:
        for j in i:
            all += 1
            if abs(j) >= 0.5: count += 1

    print(count, all)
    
    plt.figure(figsize=(15, 8))
    n_sessions = len(FR1)
    bar_width = 0.5

    x_pos = np.arange(n_sessions)
    plt.bar(x_pos, [np.mean(np.abs(exp)) for exp in correlation_coeffs], width=bar_width, alpha=0.7)

    for m, mouse in enumerate(mice):
        plt.scatter(x_pos, mean_correlation[m, :], color="black", s=5)
                
    plt.xlabel("Session")
    plt.ylabel("Correlation value (pearson)")
    plt.xticks(np.arange(n_sessions), labels, rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, coeffs in enumerate(correlation_coeffs):
        row = i // 3  # 0 pour les 3 premiers, 1 pour les 3 suivants
        col = i % 3   # 0, 1, 2 pour chaque ligne

        axes[row, col].hist(coeffs, bins=20, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'{labels[i]} (n = {len(coeffs)})')
        axes[row, col].set_xlabel('Pearson correlation coefficient')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True, alpha=0.3)

        # Optionnel : ajouter la moyenne
        axes[row, col].axvline(np.mean(coeffs), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(coeffs):.3f}')
        axes[row, col].legend()

    plt.tight_layout()
    plt.show()

    # for i in range(len(neurons)-1):
    #     trace1 = traces[neurons[i]]
    #     for j in range(i+1, len(neurons)):
    #         trace2 = traces[neurons[j]]
    #         corr = np.correlate(trace1 - np.mean(trace1), trace2 - np.mean(trace2), mode='full')

    #     lags = np.arange(-len(trace1) + 1, len(trace1))

    # trace1 = traces['C38']
    # trace2 = traces['C42']

    # corr = np.correlate(trace1 - np.mean(trace1), trace2 - np.mean(trace2), mode='full')
    # lags = np.arange(-len(trace1) + 1, len(trace1))
    
    # # Plot the cross-correlation
    # plt.figure(figsize=(10, 6))
    # plt.plot(lags, corr)
    # plt.axvline(x=0, color='r', linestyle='--')  # Add vertical line at zero lag
    # # plt.title(f'Cross-correlation between {neurons[0]} and {neurons[1]}')
    # plt.xlabel('Lag')
    # plt.ylabel('Correlation')
    # plt.grid(True)
    # plt.show()

    # trace1 = traces['C259']
    # trace2 = traces['C277']

    # corr = np.correlate(trace1 - np.mean(trace1), trace2 - np.mean(trace2), mode='full')
    # lags = np.arange(-len(trace1) + 1, len(trace1))
    
    # # Plot the cross-correlation
    # plt.figure(figsize=(10, 6))
    # plt.plot(lags, corr)
    # plt.axvline(x=0, color='r', linestyle='--')  # Add vertical line at zero lag
    # # plt.title(f'Cross-correlation between {neurons[0]} and {neurons[1]}')
    # plt.xlabel('Lag')
    # plt.ylabel('Correlation')
    # plt.grid(True)

    # trace1 = traces['C252']
    # trace2 = traces['C261']

    # corr = np.correlate(trace1 - np.mean(trace1), trace2 - np.mean(trace2), mode='full')
    # lags = np.arange(-len(trace1) + 1, len(trace1))
    
    # # Plot the cross-correlation
    # plt.figure(figsize=(10, 6))
    # plt.plot(lags, corr)
    # plt.axvline(x=0, color='r', linestyle='--')  # Add vertical line at zero lag
    # # plt.title(f'Cross-correlation between {neurons[0]} and {neurons[1]}')
    # plt.xlabel('Lag')
    # plt.ylabel('Correlation')
    # plt.grid(True)
    # plt.show()
    
    # You can also visualize the correlation matrices as heatmaps
    # plt.figure(figsize=(12, 5))
 
    # plt.subplot(1, 2, 1)
    # sns.heatmap(pearson_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    # plt.title('Pearson Correlation')

    # plt.tight_layout()
    # plt.show()

    return

def cross_correlation():
    mice = ["M2", "M4", "M15"]
    labels = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "6th FR1"]
    FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]

    all_lags = []
    max_t = 0
    for i, mouse in enumerate(mice):
        folder = f"neuronal_activity_data\\calcium_traces\\{mouse}\\status_based_2"
        FR1_file_list = [os.path.join(folder, f) for f in os.listdir(folder) if any(exp in f for exp in FR1)]
        for j, f in enumerate(FR1_file_list):
            traces = pd.read_csv(f)
            neurons = traces.columns[1:]
            new_t = np.max(traces["Time"])
            max_t = np.max([max_t, new_t])
            for r in range(len(neurons)-1):
                    for c in range(r+1, len(neurons)):
                        trace1 = traces[neurons[r]]
                        trace2 = traces[neurons[c]]
                        corr = np.correlate(trace1, trace2, "full")
                        max_corr = np.argmax(np.abs(corr))
                        lags = np.arange(-len(trace1) + 1, len(trace1))
                        lag = lags[max_corr]
                        all_lags.append(lag)

    plt.figure(figsize=(12, 6))
    
    plt.hist(all_lags, bins=1200, alpha=0.7, edgecolor='black', density=True)
    plt.xlabel('Lag (samples)')
    plt.ylabel('Probability density')
    plt.grid(True, alpha=0.3)
    
    # Ajouter des statistiques
    plt.axvline(np.mean(all_lags), color='red', linestyle='--', 
               label=f'Mean: {np.mean(all_lags):.1f}')
    plt.axvline(np.median(all_lags), color='orange', linestyle='--', 
               label=f'Median: {np.median(all_lags):.1f}')
    plt.axvline(0, color='green', linestyle='-', alpha=0.5, 
               label='Lag = 0 (synchrone)')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def correlation_during_behavior(behavior):
    mice = ["M2", "M4", "M15"]
    labels = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "6th FR1"]
    FR1 = ["Exp 010", "Exp 011", "Exp 012", "Exp 013", "Exp 014", "Exp 016"]
    behavior_columns = [behavior + " Start", behavior +  " End"]

    mean_correlation = np.zeros((len(mice), len(FR1)))
    correlation_coeffs = [[] for exp in FR1]
    
    for i, mouse in enumerate(mice):
        trace_folder = f"neuronal_activity_data\\calcium_traces\\{mouse}\\status_based_2"
        trace_file_list = [os.path.join(trace_folder, f) for f in os.listdir(trace_folder) if any(exp in f for exp in FR1)]
        behavior_folder = f"behavioral_data\\behavior descriptions\\final_description\\{mouse}"
        behavior_file_list = [os.path.join(behavior_folder, f) for f in os.listdir(behavior_folder) if any(exp in f for exp in FR1)]

        for j, files in enumerate(zip(trace_file_list, behavior_file_list)):
            traces = pd.read_csv(files[0])
            behavior_data = pd.read_csv(files[1])[behavior_columns].dropna()
            max_t = np.max(traces["Time"])
            if (not traces.empty) & (not behavior_data.empty):
                neurons = traces.columns[1:]
                behavior_traces = []
                for _, row in behavior_data.iterrows():
                    start = row[behavior_columns[0]]/1000
                    end = row[behavior_columns[1]]/1000

                    if end <= max_t:
                        behavior_segment = traces[(traces["Time"] >= start) & (traces["Time"] <= end)][neurons]
                        behavior_traces.append(behavior_segment)
                
                if behavior_traces:
                    combined_traces = pd.concat(behavior_traces, ignore_index=True)

                    pearson_corr = combined_traces[neurons].corr('pearson')
                    top_triangle = np.triu(pearson_corr, 1)
                    mean_abs = np.mean(np.abs(top_triangle), where=((top_triangle != 0)))

                    if np.isnan(mean_abs):
                        mean_abs = 0
                    mean_correlation[i, j] = mean_abs

                    for r in range(len(neurons)-1):
                        for c in range(r+1, len(neurons)):
                            trace1 = combined_traces[neurons[r]]
                            trace2 = combined_traces[neurons[c]]
                            coeff = np.corrcoef(trace1, trace2)[0, 1]

                            correlation_coeffs[j].append(coeff)

    all = 0
    count = 0
    for i in correlation_coeffs:
        for j in i:
            all += 1
            if abs(j) >= 0.5: count += 1

    print(count, all)
    
    plt.figure(figsize=(15, 8))
    n_sessions = len(FR1)
    bar_width = 0.5

    x_pos = np.arange(n_sessions)
    plt.bar(x_pos, [np.mean(np.abs(exp)) for exp in correlation_coeffs], width=bar_width, alpha=0.7)

    for m, mouse in enumerate(mice):
        plt.scatter(x_pos, mean_correlation[m, :], color="black", s=5)
                
    plt.xlabel("Session")
    plt.ylabel("Correlation value (pearson)")
    plt.xticks(np.arange(n_sessions), labels, rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, coeffs in enumerate(correlation_coeffs):
        row = i // 3  # 0 pour les 3 premiers, 1 pour les 3 suivants
        col = i % 3   # 0, 1, 2 pour chaque ligne

        axes[row, col].hist(coeffs, bins=20, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'{labels[i]} (n = {len(coeffs)})')
        axes[row, col].set_xlabel('Pearson correlation coefficient')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].grid(True, alpha=0.3)

        # Optionnel : ajouter la moyenne
        axes[row, col].axvline(np.mean(coeffs), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(coeffs):.3f}')
        axes[row, col].legend()

    plt.tight_layout()
    plt.show()



def main():
    # activity_behaviors("density")
    # correlation()
    # cross_correlation()
    correlation_during_behavior("Pressing")
    return

if __name__ == "__main__":
    main() 