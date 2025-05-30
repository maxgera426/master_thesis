import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, integrate

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

def n_press_before_drinking(data):
    first_licks = get_first_lick(data)

    full_first_licks = first_licks[first_licks["type"] == 1]
    empty_first_licks = first_licks[first_licks["type"] == 0]
    
    presses = data["Pressing Start"]

    n_presses_full = []
    n_presses_empty = []
    n_presses = []

    previous_t  = 0
    for _, row in first_licks.iterrows():
        t = row["time"]
        subset_presses = presses[(presses >= previous_t) & (presses <= t)]
        n_press = len(subset_presses)

        n_presses.append(n_press)

        if row["type"] == 1:
            n_presses_full.append(n_press)
        else:
            n_presses_empty.append(n_press)

        previous_t = t
    return n_presses


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

def pre_post_stats(data):
    df = pd.DataFrame(data)

    df['change_full'] = df['post_full'] - df['pre_full']
    df['change_empty'] = df['post_empty'] - df['pre_empty']
    
    t_stat_changes, p_val_changes = stats.ttest_rel(df['change_full'], df['change_empty'])
    
    print(f"Change during FULL events: {df['change_full'].mean():.3f} ± {df['change_full'].std():.3f}")
    print(f"Change during EMPTY events: {df['change_empty'].mean():.3f} ± {df['change_empty'].std():.3f}")
    print(f"\nPaired t-test (change_full vs change_empty):")
    print(f"t-statistic: {t_stat_changes:.3f}")
    print(f"p-value: {p_val_changes:.4f}")
    
    if p_val_changes < 0.05:
        print("✓ Significant difference in neural response between full and empty events")
    else:
        print("✗ No significant difference in neural response between full and empty events")
    
    t_stat_full, p_val_full = stats.ttest_rel(df['post_full'], df['pre_full'])
    t_stat_empty, p_val_empty = stats.ttest_rel(df['post_empty'], df['pre_empty'])
    
    print(f"\nWithin-event changes:")
    print(f"FULL events (post vs pre): t={t_stat_full:.3f}, p={p_val_full:.4f}")
    print(f"EMPTY events (post vs pre): t={t_stat_empty:.3f}, p={p_val_empty:.4f}")
    
    t_stat_post, p_val_post = stats.ttest_rel(df['post_full'], df['post_empty'])
    print(f"\nDirect comparison of post-event activity:")
    print(f"Post-full vs Post-empty: t={t_stat_post:.3f}, p={p_val_post:.4f}")
    
    t_stat_pre, p_val_pre = stats.ttest_rel(df['pre_full'], df['pre_empty'])
    print(f"Pre-full vs Pre-empty: t={t_stat_pre:.3f}, p={p_val_pre:.4f}")
    print("(This should NOT be significant - pre-event activity should be similar)")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0,0].scatter(df['change_full'], df['change_empty'], alpha=0.6, s=50)
    axes[0,0].plot([-2, 6], [-2, 6], 'r--', alpha=0.7, label='Equal change line')
    axes[0,0].set_xlabel('Change in Full Events (post - pre)')
    axes[0,0].set_ylabel('Change in Empty Events (post - pre)')
    axes[0,0].set_title('Neural Response Changes: Full vs Empty')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    corr_coef = np.corrcoef(df['change_full'], df['change_empty'])[0,1]
    axes[0,0].text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=axes[0,0].transAxes,
                   bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.8))
    
    change_data = [df['change_full'], df['change_empty']]
    bp = axes[0,1].boxplot(change_data, tick_labels=['Full Events', 'Empty Events'], 
                           patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[0,1].set_ylabel('Change in Neural Activity (post - pre)')
    axes[0,1].set_title('Distribution of Neural Response Changes')
    axes[0,1].grid(True, alpha=0.3)
    
    if p_val_changes < 0.05:
        y_max = max(df[['change_full', 'change_empty']].max())
        axes[0,1].plot([1, 2], [y_max + 0.5, y_max + 0.5], 'k-', linewidth=1)
        axes[0,1].text(1.5, y_max + 0.7, f'p = {p_val_changes:.4f}*', ha='center', fontweight='bold')
    
    axes[1,0].plot([0.8, 1.2], [df['pre_full'].mean(), df['post_full'].mean()], 
                   'bo-', linewidth=2, markersize=8, label='Mean')
    for i in range(min(20, len(df))):
        axes[1,0].plot([0.8, 1.2], [df.iloc[i]['pre_full'], df.iloc[i]['post_full']], 
                       'b-', alpha=0.3, linewidth=0.5)
    axes[1,0].set_xlim(0.5, 1.5)
    axes[1,0].set_xticks([0.8, 1.2])
    axes[1,0].set_xticklabels(['Pre-Full', 'Post-Full'])
    axes[1,0].set_ylabel('Neural Activity')
    axes[1,0].set_title(f'Full Events: Pre vs Post\n(t={t_stat_full:.2f}, p={p_val_full:.4f})')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot([0.8, 1.2], [df['pre_empty'].mean(), df['post_empty'].mean()], 
                   'ro-', linewidth=2, markersize=8, label='Mean')
    for i in range(min(20, len(df))):  # Show first 20 neurons
        axes[1,1].plot([0.8, 1.2], [df.iloc[i]['pre_empty'], df.iloc[i]['post_empty']], 
                       'r-', alpha=0.3, linewidth=0.5)
    axes[1,1].set_xlim(0.5, 1.5)
    axes[1,1].set_xticks([0.8, 1.2])
    axes[1,1].set_xticklabels(['Pre-Empty', 'Post-Empty'])
    axes[1,1].set_ylabel('Neural Activity')
    axes[1,1].set_title(f'Empty Events: Pre vs Post\n(t={t_stat_empty:.2f}, p={p_val_empty:.4f})')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    pooled_std = np.sqrt((df['change_full'].var() + df['change_empty'].var()) / 2)
    cohens_d = (df['change_full'].mean() - df['change_empty'].mean()) / pooled_std
    
    print(f"Cohen's d (change_full vs change_empty): {cohens_d:.3f}")
    if abs(cohens_d) < 0.2:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    print(f"Effect size: {effect_size}")
    
    print(f"""
    Your research question: "Is there a difference in neural activity between full and empty drinking events?"
    
    The paired t-test your supervisor suggested compares:
    - How much neural activity changes during FULL events (post_full - pre_full)
    - How much neural activity changes during EMPTY events (post_empty - pre_empty)
    
    Key results:
    1. Average change during FULL events: {df['change_full'].mean():.3f}
    2. Average change during EMPTY events: {df['change_empty'].mean():.3f}
    3. Statistical difference: p = {p_val_changes:.4f}
    4. Effect size (Cohen's d): {cohens_d:.3f} ({effect_size})
    
    Biological interpretation:
    {'✓ Neurons respond DIFFERENTLY to full vs empty drinking events' if p_val_changes < 0.05 else '✗ No significant difference in neural response between event types'}
    """)


def plot_activity(full_activity, empty_activity, t):
    x = np.linspace(-t, t, len(full_activity[0,:,0]))
    for i in range(len(full_activity[0,0,:])):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Calculate means and stds
        mean_reward = np.mean(full_activity[:,:,i], axis=0)
        std_reward = np.std(full_activity[:,:,i], axis=0)
        mean_omission = np.mean(empty_activity[:,:,i], axis=0)
        std_omission = np.std(empty_activity[:,:,i], axis=0)

        # Plot full activity (reward) on left subplot
        for trace in full_activity[:,:,i]:
            ax1.plot(x, trace, alpha=0.2, color="blue")
        ax1.plot(x, mean_reward, color="blue", label="Reward", linewidth=2)
        # ax1.fill_between(x, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2, color='b')
        ax1.axvline(x=0, color='k', linestyle='--', label='Lick event')
        ax1.set_title('Reward Activity')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Activity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot empty activity (omission) on right subplot
        for trace in empty_activity[:,:,i]:
            ax2.plot(x, trace, alpha=0.2, color="red")
        ax2.plot(x, mean_omission, color="red", label="Omission", linewidth=2)
        # ax2.fill_between(x, mean_omission - std_omission, mean_omission + std_omission, alpha=0.2, color='r')
        ax2.axvline(x=0, color='k', linestyle='--', label='Lick event')
        ax2.set_title('Omission Activity')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Activity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        
    
def main():
    # NUMBER OF PRESSES BEFORE VISITING TROUGH
    mice = ["M2", "M4", "M15"]
    press_dict = {
        "Fifth FR1": [],
        "Random omission": []
    }
    for mouse in mice:
        folder = f"behavioral_data\\behavior descriptions\\final_description\\{mouse}"
        file_list = [os.path.join(folder, f) for f in os.listdir(folder)][4:6]
        fifth_data = pd.read_csv(file_list[0])
        random_data = pd.read_csv(file_list[1])

        fifth_presses = n_press_before_drinking(fifth_data)
        random_presses = n_press_before_drinking(random_data)

        press_dict["Fifth FR1"].extend(fifth_presses)
        press_dict["Random omission"].extend(random_presses)

    
    # plt.figure()
    # sns.boxplot(press_dict)
    # plt.ylabel("Number of presses")
    # plt.xlabel("Session")
    # plt.show()

    colors = ["lightcoral", "lightblue"]

    fig, ax =plt.subplots(1,2)
    for i, item in enumerate(press_dict.items()):
        name = item[0]
        values = item[1]
        x, counts = np.unique(values, return_counts=True)
        ax[i].bar(x, counts/len(values), width=1, color=colors[i], edgecolor="black")
        ax[i].set_ylim(0,1)
        ax[i].set_xticks(range(np.max(values) + 1))
        ax[i].set_xlabel("Number of presses")
        ax[i].set_title(name)
    ax[0].set_ylabel("Relative frequency")
    plt.show()

    # # ACTIVITY BEFORE AND AFTER LICK OR PRESS EVENT
    # t = 1 # time before and after the event
    # size = int(2*t/0.05 + 1) # number of points in total
    # mice = ["M2", "M4", "M15"]

    # pre_post_activity = {
    #     "pre_full": [],
    #     "post_full": [],
    #     "pre_empty": [],
    #     "post_empty": []
    # }

    # for mouse in mice:
    #     behavior_folder = f"behavioral_data\\behavior descriptions\\final_description\\{mouse}\\"
    #     behavior_file = [os.path.join(behavior_folder, f) for f in os.listdir(behavior_folder)][5]
        
    #     neuronal_folder = f"neuronal_activity_data\\calcium_traces\\{mouse}\\status_based_2"
    #     neuron_file = [os.path.join(neuronal_folder, f) for f in os.listdir(neuronal_folder)][5]

    #     behavior_data = pd.read_csv(behavior_file)
    #     n_press_before_drinking(behavior_data)
    #     traces = pd.read_csv(neuron_file)
    #     neurons = traces.columns[1:]

    #     first_licks = get_first_lick(behavior_data)
    #     full_first_licks = first_licks[first_licks["type"] == 1]
    #     empty_first_licks = first_licks[first_licks["type"] == 0]

    #     full_activity = activity_around(traces, full_first_licks["time"], size)
    #     empty_activity = activity_around(traces, empty_first_licks["time"], size)

    #     # plot_activity(full_activity, empty_activity, t)

    #     mean_full_activity = np.mean(full_activity, axis=0)
    #     mean_empty_activity = np.mean(empty_activity, axis=0)

    #     for i in range(len(neurons)):
    #         full_neuron_activity = mean_full_activity[:,i]
    #         empty_neuron_activity = mean_empty_activity[:,i]
            
    #         pre_full = full_neuron_activity[:size//2]
    #         post_full = full_neuron_activity[size//2:]
            
    #         pre_empty = empty_neuron_activity[:size//2]
    #         post_empty = empty_neuron_activity[size//2:]

    #         pre_post_activity["pre_full"].append(np.trapezoid(pre_full))
    #         pre_post_activity["post_full"].append(np.trapezoid(post_full))
    #         pre_post_activity["pre_empty"].append(np.trapezoid(pre_empty))
    #         pre_post_activity["post_empty"].append(np.trapezoid(post_empty))
            
    # activity_data = pd.DataFrame(pre_post_activity)
    
    # pre_post_stats(activity_data)

    return

if __name__ == "__main__":
    main()