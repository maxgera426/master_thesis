import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import percentileofscore
from time import time

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
        fig = plt.figure(figsize=(10, 6))
        plt.plot(cell_traces["Time"], trace)

        # Calculate statistics
        mean = np.mean(trace)
        std = np.std(trace)
        min_h = mean + 3 * std

        # Draw horizontal lines
        plt.hlines(min_h, 0, np.max(cell_traces["Time"]), colors='g', linestyles="dashed", label="Min h")
        plt.hlines(mean, 0, np.max(cell_traces["Time"]), colors='r', linestyles="dashed", label="Mean")

        # Find peaks
        peak_indices, peak_properties = find_peaks(trace, height=min_h, prominence=min_h/3)
        peak_times = cell_traces["Time"][peak_indices]
        peak_heights = trace[peak_indices].values

        # Calculate prominences
        prominences, left_bases, right_bases = peak_prominences(trace, peak_indices)

        # Plot peaks
        plt.scatter(peak_times, peak_heights, c='r', marker='x', label="Peaks")

        # Draw prominence for each peak
        for i, (idx, prominence) in enumerate(zip(peak_indices, prominences)):
            if i == 0:
                plt.vlines(x=cell_traces["Time"][idx], ymin=peak_heights[i]-prominence, 
                       ymax=peak_heights[i], colors='b', linestyles='dashed', label= "Prominence")
            else:
                plt.vlines(x=cell_traces["Time"][idx], ymin=peak_heights[i]-prominence, 
                       ymax=peak_heights[i], colors='b', linestyles='dashed')

        plt.xlim(490, 570)
        plt.ylabel("Signal intensity")

    else:
        fig, axs = plt.subplots(n_cells, 1, figsize=(10, 10), sharex=True)
        for i, cell in enumerate(acc_cells):
            axs[i].plot(cell_traces["Time"], cell_traces[cell])
            axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            axs[i].set_ylabel(cell, rotation=0, labelpad=20)
            fig.suptitle("Neuronal Traces", fontweight= 'bold')
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()

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
                freqs.append(peak_freq_behavior(peak_times, behavior_times))
            freqs = np.array(freqs)
            print(f"Cell: {cell}, Behavior freqs: {freqs}")
    return 0

def peak_freq_behavior(peak_times, behavior_times):
    occ = 0
    total_time = np.sum(behavior_times[:, 1] - behavior_times[:, 0])/1000
    for start, end in behavior_times:
        occ += np.sum((peak_times>= start) & (peak_times <= end))
    return occ/total_time

def mean_freq_behavior(time, trace, behavior_times):
    total_time = np.sum(behavior_times[:, 1] - behavior_times[:, 0])/1000
    all_values = []
    indices = np.argsort(time)
    sorted_time = time[indices]
    sorted_trace = trace[indices]

    for start, end in behavior_times:
        # Use searchsorted for faster index finding
        start_idx = np.searchsorted(sorted_time, start/1000)
        end_idx = np.searchsorted(sorted_time, end/1000, side='right')
        
        if start_idx < end_idx:
            window_values = sorted_trace[start_idx:end_idx]
            all_values.append(window_values)
    
    if all_values:
        all_values = np.concatenate(all_values)
        overall_mean = np.mean(all_values)
    else:
        overall_mean = np.nan 
    
    return overall_mean/total_time

def density_freq_behavior(peak_times, behavior_times, total_density):
    occ = 0
    total_time = np.sum(behavior_times[:, 1] - behavior_times[:, 0])/1000
    for start, end in behavior_times:
        occ += np.sum((peak_times>= start) & (peak_times <= end))
    density = occ/total_time
    return density/total_density


def shift_peaks(peak_times, offset, total_t):
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
        save_file = r"neuronal_activity_data\calcium_traces\M15\status_based_2\\" + exp_num + "_" + os.path.splitext(os.path.basename(file_path))[0] + "_accepted_traces.csv"
        acc_data.to_csv(save_file, index=False)

def behavioral_activity_peaks(behavior_file, behavior_list, peak_times, max_t):
    offsets = np.array(list(range(-1000, -99)) + list(range(100, 1001)))*1000
    offsets = offsets/10
    
    behaviors_description = pd.read_csv(behavior_file)
    behavior_windows = {}
    for behavior in behavior_list:
        behavior_windows[behavior] = behaviors_description[[behavior + " Start", behavior + " End"]][behaviors_description[behavior + " End"] <= max_t].dropna().values
    
    org_freqs = np.array([peak_freq_behavior(peak_times, behavior_windows[behavior]) if behavior_windows[behavior].any() else np.nan for behavior in behavior_list])
    
    offset_frequencies = np.zeros((len(offsets), len(behavior_list)))
    for i, offset in enumerate(offsets):
        if not offset%5000:
            print(offset/1000)
        shifted_peaks = np.mod(peak_times + offset, max_t)
        shifted_peaks.sort()
        
        for j, behavior in enumerate(behavior_list):
            offset_frequencies[i, j] = peak_freq_behavior(shifted_peaks, behavior_windows[behavior])
    
    neuron_activity = []
    for i in range(len(behavior_list)):
        frequencies = offset_frequencies[:, i]
        org_freq = org_freqs[i]

        if np.isnan(org_freq): 
                neuron_activity[i].append(np.nan)
                continue
        
        min_freq = np.percentile(frequencies, 5)
        max_freq = np.percentile(frequencies, 95) 

        if org_freq > max_freq:
            freq_category = 1
        elif org_freq < min_freq:
            freq_category = -1
        else:
            freq_category = 0
        
        neuron_activity.append(freq_category)

    return neuron_activity

def behavioral_activity_mean(behavior_file, behavior_list, cell_traces, max_t):
    offsets = np.array(list(range(-1000, -99)) + list(range(100, 1001)))
    offsets = offsets/10
    behaviors_description = pd.read_csv(behavior_file)
    behavior_windows = {}
    cells = cell_traces.columns[1:]
    org_freqs = [[] for cell in cells]
    time_vector = np.array(cell_traces["Time"].values)
    cell_values = {cell: cell_traces[cell].values for cell in cells}

    for behavior in behavior_list:
        behavior_windows[behavior] = behaviors_description[[behavior + " Start", behavior + " End"]][behaviors_description[behavior + " End"] <= max_t].dropna().values

    for i, cell in enumerate(cells) : 
        trace = cell_traces[cell]
        org_freqs[i] = np.array([mean_freq_behavior(time_vector, trace, behavior_windows[behavior]) if behavior_windows[behavior].any() else np.nan for behavior in behavior_list])

    offset_frequencies = np.zeros((len(offsets), len(cells), len(behavior_list)))
    
    for i, offset in enumerate(offsets):
        if not offset % 5:
            print(offset)
        shifted_time = np.mod(time_vector + offset, max_t/1000)
        shifted_time[-1] -= 1e-10

        for j, cell in enumerate(cells):
            cell_trace = cell_values[cell]
            for k, behavior in enumerate(behavior_list):
                if behavior_windows[behavior].any():
                    freq = mean_freq_behavior(shifted_time, cell_trace, behavior_windows[behavior])
                else : 
                    freq = np.nan

                offset_frequencies[i, j, k] = freq

    neuron_activity = [[] for cell in cells]
    for i in range(len(cells)):
        for j in range(len(behavior_list)):
            frequencies = offset_frequencies[:, i, j]
            org_freq = org_freqs[i][j]

            if np.isnan(org_freq): 
                neuron_activity[i].append(np.nan)
                continue

            min_freq = np.percentile(frequencies, 5)
            max_freq = np.percentile(frequencies, 95) 

            if org_freq > max_freq:
                freq_category = 1
            elif org_freq < min_freq:
                freq_category = -1
            else:
                freq_category = 0

            neuron_activity[i].append(freq_category)
    
    return neuron_activity

def behavioral_activity_density(behavior_file, behavior_list, cell_traces, max_t):
    offsets = np.array(list(range(-1000, -99)) + list(range(100, 1001)))*1000
    offsets = offsets/10
    max_t_s = max_t/1000
    behaviors_description = pd.read_csv(behavior_file)
    behavior_windows = {}

    peak_dict = detect_peaks(cell_traces)
    peak_times = peak_dict.values()
    total_density = [len(peaks)/max_t_s for peaks in peak_times]

    for behavior in behavior_list:
        behavior_windows[behavior] = np.array(behaviors_description[[behavior + " Start", behavior + " End"]][behaviors_description[behavior + " End"] <= max_t].dropna().values)
    
    org_freqs = [[] for cell in peak_times]
    offset_frequencies = np.zeros((len(offsets), len(peak_times), len(behavior_list)))
    for i, peaks in enumerate(peak_times):
        org_freqs[i] = [density_freq_behavior(peaks, behavior_windows[behavior], total_density[i]) if behavior_windows[behavior].any() else np.nan for behavior in behavior_list]

        for j, offset in enumerate(offsets):
            if not offset%5000:
                print(offset/1000)
            shifted_peaks = np.mod(peaks + offset, max_t)
            shifted_peaks.sort()

            for k, behavior in enumerate(behavior_list):
                offset_frequencies[j, i, k] = density_freq_behavior(shifted_peaks, behavior_windows[behavior], total_density[i])
    
    neuron_activity = [[] for cell in peak_times]
    for i in range(len(peak_times)):
        for j in range(len(behavior_list)):
            frequencies = offset_frequencies[:, i, j]
            org_freq = org_freqs[i][j]

            if np.isnan(org_freq): 
                neuron_activity[i].append(np.nan)
                continue

            min_freq = np.percentile(frequencies, 5)
            max_freq = np.percentile(frequencies, 95) 

            if org_freq > max_freq:
                freq_category = 1
            elif org_freq < min_freq:
                freq_category = -1
            else:
                freq_category = 0

            neuron_activity[i].append(freq_category)
    
    return neuron_activity, peak_dict


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
    
    org_freqs = np.array([peak_freq_behavior(peak_times, movement_windows[behavior]) for behavior in behavior_list])

    offset_frequencies = np.zeros((len(offsets), len(behavior_list)))
    for i, offset in enumerate(offsets):
        # Vectorized shift operation
        shifted_peaks = np.mod(peak_times + offset, max_t)
        shifted_peaks.sort()  # Sort in-place
        
        for j, behavior in enumerate(behavior_list):
            offset_frequencies[i, j] = peak_freq_behavior(shifted_peaks, movement_windows[behavior])
    
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
            continue
    return peaks_dict

def save_behavior_percentiles(neuron_file_path, behavior_file_path):
    cell_traces = pd.read_csv(neuron_file_path)
    max_t = np.max(cell_traces["Time"])*1000
    behavior_list = ['Sequence', 'Moving To Trough', 'Locomotion To Trough', 'Drinking Full', 'Moving To Lever', 'Locomotion To Lever', 'Drinking Empty', 'Off Task']

    ## COMPUTE NEURON ACTIVITY BASED ON PEAKS / BEHAVIOR
    # peaks = detect_peaks(cell_traces)
    # save_df = pd.DataFrame(columns=["Cell"] + behavior_list)

    # for cell, peak_list in peaks.items():
    #     neuron_activity = behavioral_activity_peaks(behavior_file_path, behavior_list, peak_list, max_t)
    #     print(cell, neuron_activity)
    # #     new_row = pd.Series([cell] + neuron_activity, index=["Cell"] + behavior_list)
    #     save_df = pd.concat([save_df, pd.DataFrame([new_row])], ignore_index=True)
    # exp_num = os.path.basename(neuron_file_path)[:7]
    # save_name = r"neuronal_activity_data\percentiles\M15\behavior_related\peaks_per_behavior\\" + exp_num + "_activity_peaks.csv"
    # save_df.to_csv(save_name, index=False)
    
    ## COMPUTE NEURON ACTIVITY BASED ON MEAN VALUE DURING BEHAVIOR
    save_df = pd.DataFrame(columns=["Cell"] + behavior_list)
    neuron_activity = behavioral_activity_mean(behavior_file_path, behavior_list, cell_traces, max_t)
    cells = cell_traces.columns[1:]
    for i, cell in enumerate(cells):
        new_row = pd.Series([cell] + neuron_activity[i], index=["Cell"] + behavior_list)
        save_df = pd.concat([save_df, pd.DataFrame([new_row])], ignore_index=True)
    
    exp_num = os.path.basename(neuron_file_path)[:7]
    save_name = r"neuronal_activity_data\percentiles\M15\behavior_related\mean_per_behavior\\" + exp_num + "_activity_mean.csv"
    save_df.to_csv(save_name, index=False)

    ## COMPUTE NEURON ACTIVITY BASED ON RELATIVE PEAK DENSITY
    # save_df = pd.DataFrame(columns=["Cell"] + behavior_list)
    # neuron_activity, peaks = behavioral_activity_density(behavior_file_path, behavior_list, cell_traces, max_t)
    # cells = cell_traces.columns[1:]
    # for i, cell in enumerate(peaks.keys()):
    #     new_row = pd.Series([cell] + neuron_activity[i], index=["Cell"] + behavior_list)
    #     save_df = pd.concat([save_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # exp_num = os.path.basename(neuron_file_path)[:7]
    # save_name = r"neuronal_activity_data\percentiles\M15\behavior_related\density_per_behavior\\" + exp_num + "_activity_density.csv"
    # save_df.to_csv(save_name, index=False)

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
    save_name = r"neuronal_activity_data\percentiles\M4\movement_related\\" + exp_num + "_percentiles.csv"
    save_df.to_csv(save_name, index=False)
    return

def info_about_traces(neuron_file_path):
    data = pd.read_csv(neuron_file_path)
    cells = data.columns[1:]
    if len(cells) == 0:
        return
    cell_traces = np.array(data[cells])
    max_t = np.max(data.iloc[:, 0])
    print(max_t)
    means = np.mean(cell_traces, axis=0)
    std = np.std(cell_traces, axis=0)
    peak_dict = detect_peaks(data)

    n_peaks = []
    for cell in cells:
        if cell in peak_dict.keys():
            n_peaks.append(len(peak_dict[cell]))
        else : 
            n_peaks.append(0)

    info = {
        'cell': cells,
        'mean': means,
        'std': std,
        'n peaks': n_peaks,
        'density': n_peaks/max_t
    }

    info_df = pd.DataFrame(info)
    exp_num = os.path.basename(neuron_file_path)[:7]
    save_name = r"neuronal_activity_data\percentiles\M4\trace_info\\" + exp_num + "_trace_info.csv"
    info_df.to_csv(save_name, index=False)

    

def main():
    # trace_dir = r"neuronal_activity_data\calcium_traces\M4\status_based_2\\"
    # behavior_dir = r"behavioral_data\behavior descriptions\final_description\M4\\"
    # trace_file_list = [trace_dir + f for f in os.listdir(trace_dir)]
    # behavior_file_list = [behavior_dir + f  for f in os.listdir(behavior_dir)]
    # print(len(behavior_file_list), len(trace_file_list))

    # for neuron_file_path, behavior_file_path in zip(trace_file_list, behavior_file_list):
    #     # save_behavior_percentiles(neuron_file_path, behavior_file_path)
    #     info_about_traces(neuron_file_path)

    ### PLOT PERCENTILE GRAPH
    # neuron_file_path = r"neuronal_activity_data\calcium_traces\M2\status_based_2\Exp 010_M2_240619_FR1_1_CellTraces_accepted_traces.csv"
    # behavior_file_path = r"behavioral_data\behavior descriptions\full session\M2\M2 - Jun24_Exp 010_behavior_description.csv"
    # cell_traces = pd.read_csv(neuron_file_path)
    # max_t = np.max(cell_traces["Time"])*1000
    # peaks = detect_peaks(cell_traces)
    # cell = list(peaks.keys())[0]
    # behavior_list = ['Sequence', 'Moving To Zone 1', 'Moving To Trough', 'Drinking Full', 'Moving To Zone 2', 'Moving To Lever', 'Drinking Empty', 'Off Task']
    # peak_times = peaks[cell]
    # offsets = np.array(list(range(-1000, -99)) + list(range(100, 1001)))*1000
    # offsets = offsets/10

    # behaviors_description = pd.read_csv(behavior_file_path)
    # behavior_windows = {}
    # for behavior in behavior_list:
    #     behavior_windows[behavior] = behaviors_description[[behavior + " Start", behavior + " End"]].dropna().values

    # org_freqs = np.array([peak_freq_behavior(peak_times, behavior_windows[behavior]) for behavior in behavior_list])

    # offset_frequencies = np.zeros((len(offsets), len(behavior_list)))
    # for i, offset in enumerate(offsets):
    #     # Vectorized shift operation
    #     shifted_peaks = np.mod(peak_times + offset, max_t)
    #     shifted_peaks.sort()  # Sort in-place

    #     for j, behavior in enumerate(behavior_list):
    #         offset_frequencies[i, j] = peak_freq_behavior(shifted_peaks, behavior_windows[behavior])


    # fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    # axs = axs.flatten()

    # for j, behavior in enumerate(behavior_list):
    #     # Get frequency distribution for this behavior
    #     frequencies = offset_frequencies[:, j]

    #     # Calculate percentile of the original frequency in the distribution
    #     percentile = percentileofscore(frequencies, org_freqs[j], kind = "weak")

    #     # Create histogram
    #     counts, bins, _ = axs[j].hist(frequencies, bins=50, alpha=0.7, color='skyblue')

    #     # Plot vertical line for original frequency
    #     axs[j].axvline(x=org_freqs[j], color='red', linestyle='--', 
    #                    label=f'Original: {org_freqs[j]:.3f}\nPercentile: {percentile:.1f}%')

    #     # Set title and labels
    #     axs[j].set_title(f'{behavior}')
    #     axs[j].set_xlabel('Frequency')
    #     axs[j].set_ylabel('Count')
    #     axs[j].legend()


    # plt.tight_layout()
    # plt.suptitle('Frequency Distributions by Behavior with Original Values', y=1.02, fontsize=16)
    # plt.show()

    # percentiles = [percentileofscore(offset_frequencies[:, j], org_freqs[j]) for j in range(len(behavior_list))]

    # plt.figure(figsize=(12, 6))
    # bars = plt.bar(behavior_list, percentiles, color='skyblue')

    # for i, p in enumerate(percentiles):
    #     if p > 95:
    #         bars[i].set_color('green')
    #     elif p < 5:
    #         bars[i].set_color('red')

    #     # Add text labels
    #     plt.text(i, percentiles[i] + 2, f'{percentiles[i]:.1f}%', 
    #              ha='center', va='bottom', fontsize=10)

    # plt.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95th percentile')
    # plt.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5th percentile')
    # plt.axhline(y=50, color='black', linestyle=':', alpha=0.5, label='50th percentile')

    # plt.ylabel('Percentile (%)')
    # plt.xlabel('Behavior')
    # plt.title('Percentile of Original Frequency in Offset Distribution')
    # plt.xticks(rotation=45, ha='right')
    # plt.ylim(0, 105)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    ### TO PLOT A / MULTIPLE CELL TRACE WITH DETECTED PEAKS
    cell_traces = pd.read_csv(r"neuronal_activity_data\calcium_traces\M2\status_based_2\Exp 016_M2_240623_FR1_6_CellTraces_accepted_traces.csv")
    plot_cell_traces(cell_traces, ['C38', 'C42'])

if __name__ == "__main__":
    main()