import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats

def get_first_lick(data):
    # list of first lick times after press sequence
    end = data["Sequence End"].dropna()
    full_licks = data["Licking Full Start"]
    empty_licks = data["Licking Empty Start"]

    full_df = pd.DataFrame({
        'time': full_licks,
        'type': 1  # 1 indicates full lick
    })

    empty_df = pd.DataFrame({
        'time': empty_licks,
        'type': 0  # 0 indicates empty lick
    })

    combined_df = pd.concat([full_df, empty_df], ignore_index=True)
    combined_df = combined_df.sort_values('time').reset_index(drop=True)

    first_licks = []
    for t in end:
        next_licks = combined_df[combined_df['time'] > t]
        if not next_licks.empty :
            lick_t, l_type = next_licks.iloc[0]
            first_licks.append([lick_t, l_type])
    
    first_licks = list(map(list, set(map(tuple, first_licks))))
    
    return pd.DataFrame(first_licks, columns=["time", "type"]).sort_values(by='time').reset_index(drop=True)

def get_last_press(data):
    first_licks = get_first_lick(data)
    presses = data[["Pressing Start", "Pressing End"]]
    last_presses = []
    for t in first_licks["time"]:
        subset_presses = presses[presses["Pressing End"] <= t]
        press_start, press_end = subset_presses.iloc[-1]
        last_presses.append([press_start, press_end])
    
    return pd.DataFrame(last_presses, columns=["start", "end"]).sort_values(by="start").reset_index(drop=True)


def activity_around(traces, lick_times, size = 41):
    neurons = traces.columns[1:]

    trace_intervals = np.zeros((len(lick_times), size, len(neurons)))
    for i, t in enumerate(lick_times):
        diff = np.abs(traces["Time"] - t/1000)
        idx = np.argmin(diff)
        min_idx = int(np.max([idx - (size-1)/2, 0]))
        max_idx = int(np.min([idx + (size-1)/2 + 1, len(traces)]))

        interval = traces.iloc[min_idx:max_idx]

        if not interval.empty:
            for j, neuron in enumerate(neurons):

                trace = interval[neuron]

                for k, val in enumerate(trace):
                    trace_intervals[i,k,j] = val

    return trace_intervals

def main():

    data = pd.read_csv(r"behavioral_data\behavior descriptions\final_description\M2\M2 - Jun24_Exp 015_behavior_description.csv")
    print(get_last_press(data))


    # t = 1 # time before and after the event
    # size = int(2*t/0.05 + 1) # number of points in total
    # mice = ["M2", "M4", "M15"]
    # all_full_activity = []
    # all_empty_activity = []
    # for mouse in mice:
    #     behavior_folder = f"behavioral_data\\behavior descriptions\\final_description\\{mouse}\\"
    #     behavior_file = [os.path.join(behavior_folder, f) for f in os.listdir(behavior_folder)][5]
        
    #     neuronal_folder = f"neuronal_activity_data\\calcium_traces\\{mouse}\\status_based_2"
    #     neuron_file = [os.path.join(neuronal_folder, f) for f in os.listdir(neuronal_folder)][5]

    #     behavior_data = pd.read_csv(behavior_file)
    #     traces = pd.read_csv(neuron_file)

    #     first_licks = get_first_lick(behavior_data)
    #     full_first_licks = first_licks[first_licks["type"] == 1]
    #     empty_first_licks = first_licks[first_licks["type"] == 0]

    #     full_activity = activity_around(traces, full_first_licks["time"], size)
    #     empty_activity = activity_around(traces, empty_first_licks["time"], size)
        
    #     for neuron in full_activity:
    #         all_full_activity.append(neuron)
        
    #     for neuron in empty_activity:
    #         all_empty_activity.append(neuron)
        


    # print(all_full_activity[0].shape)
    # print(all_full_activity[7].shape)  

    # x = np.linspace(-t, t, len(full_activity[0,:,0]))
    # for i in range(len(full_activity[0,0,:])):
    #     plt.figure()
    #     mean_reward = np.mean(full_activity[:,:,i], axis=0)
    #     std_reward = np.std(full_activity[:,:,i], axis=0)
    #     mean_ommission = np.mean(empty_activity[:,:,i], axis=0)
    #     std_omission = np.std(empty_activity[:,:,i], axis=0)

    #     for trace in full_activity[:,:,i]:
    #         plt.plot(x, trace, alpha=0.1, color="blue")
    #     plt.plot(x, mean_reward, color="blue", label = "Reward")
    #     plt.fill_between(x, mean_reward - std_reward, mean_reward + std_reward, alpha = 0.2, color = 'b')

    #     for trace in empty_activity[:,:,i]:
    #         plt.plot(x, trace, alpha=0.1, color="red")
    #     plt.plot(x, mean_ommission, color="red", label = "Omission")
    #     plt.fill_between(x, mean_ommission - std_omission, mean_ommission + std_omission, alpha = 0.2, color = 'r')

    #     plt.axvline(x=0, color='k', linestyle='--', label='Lick event')

    #     plt.legend()
    #     plt.show()

    #     means_reward = [np.mean(trace) for trace in full_activity[:,:, i]]
    #     means_omission = [np.mean(trace) for trace in empty_activity[:,:,i]]

    #     global_mean_reward = np.mean(means_reward)
    #     global_std_reward = np.std(means_reward)

    #     global_mean_omission = np.mean(means_omission)
    #     global_std_omission = np.std(means_omission)

    #     plt.figure(figsize=(8, 6))
    #     barres = plt.bar([0, 1], 
    #                     [global_mean_reward, global_mean_omission],
    #                     yerr=[global_std_reward, global_std_omission],
    #                     capsize=10, width=0.6, color=['blue', 'red'])

    #     plt.xticks([0, 1], ['Récompense', 'Omission'])
    #     plt.ylabel('Signal calcique moyen (ΔF/F)')
    #     plt.title('Comparaison de l\'activité neuronale moyenne')
    #     plt.grid(True, alpha=0.3, axis='y')

    #     t_stat, p_value = stats.ttest_ind(means_reward, means_omission)
    #     plt.annotate(f'p-value = {p_value:.4f}', xy=(0.5, max([global_mean_reward, global_mean_omission])*1.1), 
    #                  ha='center', fontsize=12)

    #     plt.tight_layout()
    #     plt.show()

    return

if __name__ == "__main__":
    main()