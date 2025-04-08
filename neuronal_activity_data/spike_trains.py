import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

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
        plt.figure()
        plt.plot(cell_traces["Time"], cell_traces[acc_cells[0]])
        plt.ylabel(acc_cells[0])
    else:
        fig, axs = plt.subplots(n_cells, 1, figsize=(10, 10), sharex=True)
        for i, cell in enumerate(acc_cells):
            axs[i].plot(cell_traces["Time"], cell_traces[cell])
            axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            axs[i].set_ylabel(cell, rotation=0, labelpad=20)
            fig.suptitle("Neuronal Traces", fontweight= 'bold')
    plt.xlabel("Time (s)")
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
                freqs.append(freq_behavior(peak_times, behavior_times))
            freqs = np.array(freqs)
            print(f"Cell: {cell}, Behavior freqs: {freqs}")
    return 0

def freq_behavior(peak_times, behavior_times):
    occ = 0
    total_time = np.sum(behavior_times[:, 1] - behavior_times[:, 0])/1000
    for start, end in behavior_times:
        occ += np.sum((peak_times >= start) & (peak_times <= end))
    return occ/total_time


exp_list = [f"0{i}" for i in range(10, 11)]
folder_path = r"P:\Ca2+ Data\M4 - Jun24"

file_list = get_file_list(exp_list, folder_path)
print(file_list)
for file_path in file_list:
    data, props = load_data(file_path)
    cell_names = list(data.columns[1:])
    acc_cells = filter_by_status(props)
    # acc_cells = []
    # for cell in cell_names:
    #     trace = data[cell]
    #     pnr_christophe, pnr_mad, trace_filtree, peak_indx = pnr(trace)
    #     if pnr_christophe > 3.5 and pnr_mad > 7.5:
    #         acc_cells.append(cell)
    
    exp_num = os.path.basename(os.path.dirname(file_path))
    acc_data = data[["Time"] + acc_cells]
    save_file = r"neuronal_activity_data\calcium_traces\M4\status_based_2\\" + exp_num + "_" + os.path.splitext(os.path.basename(file_path))[0] + "_accepted_traces.csv"
    acc_data.to_csv(save_file, index=False)

# neuron_file_path = r"neuronal_activity_data\calcium_traces\M2\status_based_2\Exp 010_M2_240619_FR1_1_CellTraces_accepted_traces.csv"
# behavior_file_path = r"behavioral_data\behavior descriptions\full session\M2\M2 - Jun24_Exp 010_behavior_description.csv"
# cell_traces = pd.read_csv(neuron_file_path)
# behavior_list = ['Sequence', 'Moving To Zone 1', 'Moving To Trough', 'Drinking Full', 'Moving To Zone 2', 'Moving To Lever', 'Drinking Empty', 'Off Task']
# compute_freq(cell_traces, behavior_list, behavior_file_path)
