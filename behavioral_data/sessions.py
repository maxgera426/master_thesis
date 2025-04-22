import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_data_from_behavior_csv(file_path):
    # Loads the data from behavior description csv. This csv is saved using the above save_behavior_description(args) function
    data = pd.read_csv(file_path)
    return data


mice = ["M2", "M4", "M15"]
interval_list = []

for mouse in mice:
    dir = r"behavioral_data\behavior descriptions\full session\\" + mouse
    file_list = os.listdir(dir)
    for file in file_list:
        data = load_data_from_behavior_csv(os.path.join(dir, file))
        start_seq_time = data["Sequence Start"].dropna().tolist()
        for i in range(len(start_seq_time) - 1):
            interval = start_seq_time[i+1] - start_seq_time[i]
            interval_list.append(interval)

min_val = np.min(interval_list)
max_val = np.max(interval_list)

bin_size = 500 
nbins = (max_val - min_val)//bin_size + 1
x = np.arange(min_val, max_val, bin_size)
counts = np.zeros(int(nbins))
for value in interval_list:
    diff = np.abs(x - value)
    i = np.argmin(diff)
    counts[i] += 1

counts = counts/np.sum(counts)

plt.figure()
plt.bar(x, counts, width=bin_size, align='edge', edgecolor='black', color='green')

plt.xlabel('Time difference between 2 action sequences (ms)')
plt.ylabel('Frequency')

plt.xlim(0, 100000)
plt.tight_layout()
plt.show()


