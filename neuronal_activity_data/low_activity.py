import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def low_activity_traces():
    mice = ["M2", "M4", "M15"]
    min_density = 1/30

    for mouse in mice:
        info_folder = f"neuronal_activity_data\\percentiles\\{mouse}\\trace_info"
        trace_folder = f"neuronal_activity_data\\calcium_traces\\{mouse}\\status_based_2\\"

        info_file_list = [os.path.join(info_folder, f) for f in os.listdir(info_folder)]
        trace_file_list = [os.path.join(trace_folder, f) for f in os.listdir(trace_folder)]
        for info, trace_file in zip(info_file_list, trace_file_list):
            print(trace_file)
            data = pd.read_csv(info)
            filtered = data[data["density"] < min_density]
            if not filtered.empty:
                traces = pd.read_csv(trace_file)
                cells = filtered["cell"].values
                for cell in cells:
                    plot_cell_traces(traces, cell)

def plot_cell_traces(cell_traces, cell_name):

    trace = cell_traces[cell_name]
    fig = plt.figure(figsize=(10, 6))
    plt.plot(cell_traces["Time"], trace)
    # Calculate statistics
    mean = np.mean(trace)
    std = np.std(trace)
    min_h = mean + 3 * std
    # Draw horizontal lines
    plt.hlines(min_h, 0, np.max(cell_traces["Time"]), colors='g', linestyles="dashed", label="Min h")
    plt.hlines(mean, 0, np.max(cell_traces["Time"]), colors='r', linestyles="dashed", label="Mean")
    plt.ylabel("Signal intensity")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.show()


def main():
    low_activity_traces()

    return

if __name__ == "__main__":
    main()