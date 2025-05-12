import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from spike_trains import plot_cell_traces

def get_file_list(folder):
    return [folder + f for f in os.listdir(folder)]

mouse_list = ["M2", "M4", "M15"]
for mouse in mouse_list:
    trace_folder = f"neuronal_activity_data\\calcium_traces\\{mouse}\\status_based_2\\"
    percentile_folder = f"neuronal_activity_data\\percentiles\\{mouse}\\behavior_related\\"

    trace_file_list = get_file_list(trace_folder)
    percentile_file_list = get_file_list(percentile_folder)

    for trace_file, percentile_file in zip(trace_file_list, percentile_file_list):
        traces = pd.read_csv(trace_file)
        trace_cells = traces.columns[1:]

        percentiles = pd.read_csv(percentile_file)
        percentile_cells = percentiles["Cell"]

        if len(trace_cells) != len(percentile_cells):
            print("\nDifference in", os.path.basename(trace_file)[:7], "in mouse:", mouse)
            diff = list(set(trace_cells) - set(percentile_cells))
            print("Cells not found in percentiles:", diff)
            for cell in diff:
                plot_cell_traces(traces, [cell])
                plt.title(f"Trace from {mouse} session {os.path.basename(trace_file)[:7]}")
                plt.savefig(f"neuronal_activity_data\\undetected_peaks\\{mouse}_{os.path.basename(trace_file)[:7]}_{cell}_no_detected_peaks.png", dpi=300, bbox_inches='tight')
                plt.show()
