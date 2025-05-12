import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.stats import percentileofscore

def load_data(file_path):
    print("Processing file: ", file_path)
    props_file = os.path.splitext(file_path)[0] + "-props.csv"
    props_data = pd.read_csv(props_file)
    column_names = ["Time"] + list(props_data["Name"])
    cell_traces = pd.read_csv(file_path, skiprows=2, names=column_names, dtype=float, na_values=[" nan"])

    return cell_traces, props_data

def plot_cell_traces(cell_traces, acc_cells):
    n_cells = len(acc_cells)
    if n_cells == 1:
        trace = cell_traces[acc_cells[0]]
        plt.figure()
        plt.plot(cell_traces["Time"], trace)
        mean = np.mean(trace)
        std = np.std(trace)
        min_h = mean + 3 * std
        plt.hlines(min_h, 0, np.max(cell_traces["Time"]), colors='g', label="min_h")
        plt.hlines(np.mean(trace), 0, np.max(cell_traces["Time"]), colors='r', label="mean")
        peak_indices, _ = find_peaks(trace, height=min_h, prominence=min_h/3)
        peak_times = cell_traces["Time"][peak_indices]
        plt.scatter(peak_times, trace[peak_indices], c='r', marker='x')
        plt.ylabel(acc_cells[0])
    else:
        fig, axs = plt.subplots(n_cells, 1, figsize=(10, 10), sharex=True)
        for i, cell in enumerate(acc_cells):
            axs[i].plot(cell_traces["Time"], cell_traces[cell])
            axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            axs[i].set_ylabel(cell, rotation=0, labelpad=20)
            fig.suptitle("Neuronal Traces", fontweight= 'bold')
    plt.xlabel("Time (s)")
    plt.legend()
    # plt.show()

def get_file_list(exp_list, folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("CellTraces.csv"):
                if any(f"{exp}" in root for exp in exp_list):
                    file_list.append(os.path.join(root, file))
    return file_list

def pnr(trace):
    mean = np.mean(trace)
    std = np.std(trace)

    min_h = mean + 3 * std
    peak_indices, proprietes = find_peaks(trace, height=min_h, prominence=min_h/3)

    if len(peak_indices) > 0:
        amp = [trace[i] - mean for i in peak_indices]
        median_amp = np.median(amp)
        mean_amp = np.mean(amp)
        MAD = np.median(np.abs(trace - np.median(trace)))
        pnr_mean= mean_amp / std
        pnr_mad = median_amp/MAD
    else:
        pnr_mean = 0
        pnr_mad = 0
    
    return pnr_mean, pnr_mad, trace, peak_indices

def filter_by_status(props):
    accepted_cells = props[props["Status"] == "accepted"]["Name"].tolist()
    return accepted_cells

def compute_freq(cell_traces, behavior_list, behavior_file):
    cells = list(cell_traces.columns[1:])
    behaviors_description = pd.read_csv(behavior_file)
    for cell in cells:
        trace = cell_traces[cell]
        mean = np.mean(trace)
        std = np.std(trace)
        min_h = mean + 3 * std
        peak_indices, _ = find_peaks(trace, height=min_h, prominence=min_h/3)
        
        if len(peak_indices) > 0:
            peak_times = cell_traces["Time"][peak_indices]
            peak_times = np.array(peak_times)*1000
            freqs = []
            for behavior in behavior_list:
                behavior_times = behaviors_description[[behavior + " Start", behavior + " End"]].dropna().values
                freqs.append(freq_behavior(peak_times, behavior_times))
            freqs = np.array(freqs)
            print(f"Cell: {cell}, Behavior freqs: {freqs}")
    return 0

def freq_behavior(peak_times, behavior_times):
    occ = 0
    total_time = np.sum(behavior_times[:, 1] - behavior_times[:, 0])/1000
    for start, end in behavior_times:
        occ += np.sum((peak_times>= start) & (peak_times <= end))
    return occ/total_time

def shift_trace(peak_times, offset, total_t):
    shifted_peaks = np.mod(peak_times + offset, total_t)
    shifted_peaks.sort() 
    return shifted_peaks

def save_cell_traces(file_list):
    for file_path in file_list:
        data, props = load_data(file_path)
        cell_names = list(data.columns[1:])
        acc_cells = filter_by_status(props)

        exp_num = os.path.basename(os.path.dirname(file_path))
        acc_data = data[["Time"] + acc_cells]
        save_file = r"neuronal_activity_data\calcium_traces\M4\status_based_2\\" + exp_num + "_" + os.path.splitext(os.path.basename(file_path))[0] + "_accepted_traces.csv"
        acc_data.to_csv(save_file, index=False)

def compute_behavior_probability(behavior_file, behavior_list, peak_times, max_t):
    offsets = np.array(list(range(-1000, -99)) + list(range(100, 1001)))*1000
    offsets = offsets/10

    behaviors_description = pd.read_csv(behavior_file)
    behavior_windows = {}
    for behavior in behavior_list:
        behavior_windows[behavior] = behaviors_description[[behavior + " Start", behavior + " End"]].dropna().values
    
    org_freqs = np.array([freq_behavior(peak_times, behavior_windows[behavior]) for behavior in behavior_list])
    
    offset_frequencies = np.zeros((len(offsets), len(behavior_list)))
    for i, offset in enumerate(offsets):
        # Vectorized shift operation
        shifted_peaks = np.mod(peak_times + offset, max_t)
        shifted_peaks.sort()  # Sort in-place
        
        for j, behavior in enumerate(behavior_list):
            offset_frequencies[i, j] = freq_behavior(shifted_peaks, behavior_windows[behavior])
    
    # Calculate percentiles
    percentiles = [percentileofscore(offset_frequencies[:, i], org_freqs[i]) for i in range(len(behavior_list))]

    return percentiles

def compute_movement_probability(movement_file, peak_times, max_t):
    offsets = np.array(list(range(-1000, -99)) + list(range(100, 1001)))*1000
    offsets = offsets/10

    movement_segments = pd.read_csv(movement_file)
    movement_windows = {}
    movement_windows["Moving"] = movement_segments[['Start', 'Stop']].dropna().values*1000
    not_moving_starts = movement_segments['Stop'].values[:-1]*1000
    not_moving_stops = movement_segments['Start'].values[1:]*1000

    if movement_segments['Start'].values[0] > 0:
        not_moving_starts = np.insert(not_moving_starts, 0, 0)
        not_moving_stops = np.insert(not_moving_stops, 0, movement_segments['Start'].values[0])

    if movement_segments['Stop'].values[-1] < max_t:
        not_moving_starts = np.append(not_moving_starts, movement_segments['Stop'].values[-1])
        not_moving_stops = np.append(not_moving_stops, max_t)
    
    movement_windows["Not Moving"] = np.column_stack((not_moving_starts, not_moving_stops))
    behavior_list = ['Moving', 'Not Moving']
    
    org_freqs = np.array([freq_behavior(peak_times, movement_windows[behavior]) for behavior in behavior_list])

    offset_frequencies = np.zeros((len(offsets), len(behavior_list)))
    for i, offset in enumerate(offsets):
        # Vectorized shift operation
        shifted_peaks = np.mod(peak_times + offset, max_t)
        shifted_peaks.sort()  # Sort in-place
        
        for j, behavior in enumerate(behavior_list):
            offset_frequencies[i, j] = freq_behavior(shifted_peaks, movement_windows[behavior])
    
    # Calculate percentiles
    percentiles = [percentileofscore(offset_frequencies[:, i], org_freqs[i]) for i in range(len(behavior_list))]

    return percentiles

def detect_peaks(cell_traces):
    cells = list(cell_traces.columns[1:])
    peaks_dict = {}
    for cell in cells:
        trace = cell_traces[cell]
        mean = np.mean(trace)
        std = np.std(trace)
        min_h = mean + 3 * std
        peak_indices, _ = find_peaks(trace, height=min_h, prominence=min_h/3)

        if len(peak_indices) > 0:
            peak_times = cell_traces["Time"][peak_indices]
            peak_times = np.array(peak_times)*1000
            peaks_dict[cell] = peak_times
        else : 
            print(cell)
            print(peak_indices)
            plot_cell_traces(cell_traces, ['C67'])
    return peaks_dict

def save_behavior_percentiles(neuron_file_path, behavior_file_path):
    cell_traces = pd.read_csv(neuron_file_path)
    max_t = np.max(cell_traces["Time"])*1000
    peaks = detect_peaks(cell_traces)
    behavior_list = ['Sequence', 'Moving To Zone 1', 'Moving To Trough', 'Drinking Full', 'Moving To Zone 2', 'Moving To Lever', 'Drinking Empty', 'Off Task']
    save_df = pd.DataFrame(columns=["Cell"] + behavior_list)

    for cell, peak_list in peaks.items():
        percentiles = compute_behavior_probability(behavior_file_path, behavior_list, peak_list, max_t)
        print(cell, percentiles)
        new_row = pd.Series([cell] + percentiles, index=["Cell"] + behavior_list)
        save_df = pd.concat([save_df, pd.DataFrame([new_row])], ignore_index=True)

    exp_num = os.path.basename(neuron_file_path)[:7]
    save_name = r"neuronal_activity_data\percentiles\M15\\" + exp_num + "_percentiles.csv"
    save_df.to_csv(save_name, index=False)
    return

def save_movement_percentiles(neuron_file_path, movement_file_path):
    cell_traces = pd.read_csv(neuron_file_path)
    max_t = np.max(cell_traces["Time"])*1000
    peaks = detect_peaks(cell_traces)
    columns = ["Moving", "Not Moving"]
    save_df = pd.DataFrame(columns=["Cell"] + columns)
    behavior_list = ['Moving', 'Not Moving']

    for cell, peak_list in peaks.items():
        percentiles = compute_movement_probability(movement_file_path, peak_list, max_t)
        print(cell, percentiles)
        new_row = pd.Series([cell] + percentiles, index=["Cell"] + behavior_list)
        save_df = pd.concat([save_df, pd.DataFrame([new_row])], ignore_index=True)

    exp_num = os.path.basename(neuron_file_path)[:7]
    save_name = r"neuronal_activity_data\percentiles\M15\movement_related\\" + exp_num + "_percentiles.csv"
    save_df.to_csv(save_name, index=False)
    return

def main():
    # trace_dir = r"neuronal_activity_data\calcium_traces\M15\status_based_2\\"
    # behavior_dir = r"behavioral_data\behavior descriptions\full session\M15\\"
    # trace_file_list = [trace_dir + f for f in os.listdir(trace_dir)][6:7]
    # behavior_file_list = [behavior_dir + f  for f in os.listdir(behavior_dir)][6:7]
    # print(len(behavior_file_list), len(trace_file_list))

    # for neuron_file_path, behavior_file_path in zip(trace_file_list, behavior_file_list):
    #     save_behavior_percentiles(neuron_file_path, behavior_file_path)

    ### PLOT PERCENTILE GRAPH
    neuron_file_path = r"neuronal_activity_data\calcium_traces\M2\status_based_2\Exp 010_M2_240619_FR1_1_CellTraces_accepted_traces.csv"
    behavior_file_path = r"behavioral_data\behavior descriptions\full session\M2\M2 - Jun24_Exp 010_behavior_description.csv"
    cell_traces = pd.read_csv(neuron_file_path)
    max_t = np.max(cell_traces["Time"])*1000
    peaks = detect_peaks(cell_traces)
    cell = list(peaks.keys())[0]
    behavior_list = ['Sequence', 'Moving To Zone 1', 'Moving To Trough', 'Drinking Full', 'Moving To Zone 2', 'Moving To Lever', 'Drinking Empty', 'Off Task']
    peak_times = peaks[cell]
    offsets = np.array(list(range(-1000, -99)) + list(range(100, 1001)))*1000
    offsets = offsets/10

    behaviors_description = pd.read_csv(behavior_file_path)
    behavior_windows = {}
    for behavior in behavior_list:
        behavior_windows[behavior] = behaviors_description[[behavior + " Start", behavior + " End"]].dropna().values

    org_freqs = np.array([freq_behavior(peak_times, behavior_windows[behavior]) for behavior in behavior_list])

    offset_frequencies = np.zeros((len(offsets), len(behavior_list)))
    for i, offset in enumerate(offsets):
        # Vectorized shift operation
        shifted_peaks = np.mod(peak_times + offset, max_t)
        shifted_peaks.sort()  # Sort in-place

        for j, behavior in enumerate(behavior_list):
            offset_frequencies[i, j] = freq_behavior(shifted_peaks, behavior_windows[behavior])

    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    axs = axs.flatten()

    for j, behavior in enumerate(behavior_list):
        # Get frequency distribution for this behavior
        frequencies = offset_frequencies[:, j]

        # Calculate percentile of the original frequency in the distribution
        percentile = percentileofscore(frequencies, org_freqs[j], kind = "weak")

        # Create histogram
        counts, bins, _ = axs[j].hist(frequencies, bins=50, alpha=0.7, color='skyblue')

        # Plot vertical line for original frequency
        axs[j].axvline(x=org_freqs[j], color='red', linestyle='--', 
                       label=f'Original: {org_freqs[j]:.3f}\nPercentile: {percentile:.1f}%')

        # Set title and labels
        axs[j].set_title(f'{behavior}')
        axs[j].set_xlabel('Frequency')
        axs[j].set_ylabel('Count')
        axs[j].legend()


    plt.tight_layout()
    plt.suptitle('Frequency Distributions by Behavior with Original Values', y=1.02, fontsize=16)
    plt.show()

    # You might also want a summary plot showing all percentiles together
    percentiles = [percentileofscore(offset_frequencies[:, j], org_freqs[j]) for j in range(len(behavior_list))]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(behavior_list, percentiles, color='skyblue')

    # Highlight significant percentiles (e.g., >95% or <5%)
    for i, p in enumerate(percentiles):
        if p > 95:
            bars[i].set_color('green')
        elif p < 5:
            bars[i].set_color('red')

        # Add text labels
        plt.text(i, percentiles[i] + 2, f'{percentiles[i]:.1f}%', 
                 ha='center', va='bottom', fontsize=10)

    plt.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95th percentile')
    plt.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5th percentile')
    plt.axhline(y=50, color='black', linestyle=':', alpha=0.5, label='50th percentile')

    plt.ylabel('Percentile (%)')
    plt.xlabel('Behavior')
    plt.title('Percentile of Original Frequency in Offset Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.show()

    ### TO PLOT A / MULTIPLE CELL TRACE WITH DETECTED PEAKS
    # cell_traces = pd.read_csv(r"neuronal_activity_data\calcium_traces\M4\status_based_2\Exp 011_M4_240619_FR1_2_CellTraces_accepted_traces.csv")
    # plot_cell_traces(cell_traces, ['C37'])

if __name__ == "__main__":
    main()