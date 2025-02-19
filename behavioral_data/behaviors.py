import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import itertools
import movie_data

column_names = ['time','family', 'num', 'P', 'V', 'L', 'R', 'T','W', 'X', 'Y', 'Z']


def load_data(file_path):
    # Read specified .dat file and store into dataframe
    data = pd.read_csv(file_path, delimiter='\t', skiprows=12, header=None, names=column_names)
    return data

def pressing_actions(data):
    #receives df containing only lever press data, returns dataframes containing start and end moment of all pressing actions
    lever_press_df = data[data.iloc[:,1] == 2]

    start_buffer = []
    end_buffer = []

    if len(lever_press_df) != 0:

        for i in range(1, max(lever_press_df.iloc[:,7]) + 1):

            single_press_action = lever_press_df[(lever_press_df.iloc[:, 7] == i)]

            if len(single_press_action) > 1:
                start = single_press_action.iloc[0,:]
                end = single_press_action.iloc[1,:]

                start_buffer.append(start)
                end_buffer.append(end)
    
    start_df = pd.DataFrame(start_buffer, columns=data.columns)
    end_df = pd.DataFrame(end_buffer, columns=data.columns)

    return start_df, end_df

def get_zones(data, zone_index):
    # Receives df containing only zones data and returns df containing time values for specified zone
    subset = data[(data.iloc[:, 1] == 9) & (data.iloc[:, 4] == zone_index)]
    time_values = subset.iloc[:,0].drop_duplicates()
    return time_values

def action_sequences(data, start_pressing, end_pressing, drinking_times, start_in_task, end_in_task):
    # Receives data df and start and end df corresponding to data of start and end of lever presses. 
    # Returns df of start and end values of action sequences
    # Action sequences are cumulated times the mouse clicked on the lever without doing something else
    # NEED TO BE MODIFIED. Some of the presses are not taken into account, why?
    start_buffer = []
    end_buffer = []

    for start, end in zip(start_in_task, end_in_task):
        subset_start = start_pressing[(start_pressing.iloc[:,0] >= start) & (start_pressing.iloc[:,0] <= end)].reset_index(drop=True)

        if len(subset_start) != 0 :
            for ind, seq_start in subset_start.iterrows():
                seq_start_time = seq_start.iloc[0]
                subset_end = end_pressing[(end_pressing.iloc[:,0] >= seq_start_time) & (end_pressing.iloc[:,0] <= end)].reset_index(drop=True)
                if len(subset_end) != 0 :

                    for i in range(len(subset_end)-1, -1, -1):
                        seq_end = subset_end.iloc[i]

                        subset_drinking = drinking_times[(drinking_times >= seq_start_time) & (drinking_times <= seq_end.iloc[0])]

                        seq_start_tuple = tuple(seq_start)
                        seq_end_tuple = tuple(seq_end)

                        if subset_drinking.empty:
                            if seq_end_tuple not in end_buffer:  # Use tuple for comparison
                                start_buffer.append(seq_start_tuple)
                                end_buffer.append(seq_end_tuple) 
                            break
                        else :
                            continue
                
    seq_starts_df = pd.DataFrame(start_buffer, columns= data.columns).drop_duplicates().reset_index(drop=True)
    seq_ends_df = pd.DataFrame(end_buffer, columns= data.columns).drop_duplicates().reset_index(drop=True)

    return seq_starts_df, seq_ends_df

def enter_zone1(data, end_times):
    # This functions returns the times at which mouse enters zone 1 before starting an action sequence
    # end_times variable is a df containing the start times of action sequences
    start = 0
    buffer = []

    for i in range(len(end_times)):
        time = end_times.iloc[i,0]
        lim = data[data.iloc[:,0] == time].index[0]

        subset = data.iloc[start:lim, : ].query("family == 9 & V == 1")
        if not subset.empty:
            last_enter = subset.iloc[-1]
            buffer.append(last_enter)
        start = lim

    enter_zone1_df = pd.DataFrame(buffer, columns=data.columns)
    return enter_zone1_df

def enter_zone2(data, start_times):
    # This functions returns the times at which mouse enters zone 2 after ending an action sequence
    # start_times variable is a df containing the end times of action sequences

    buffer = []

    for i in range(len(start_times)):
        time = start_times.iloc[i,0]
        lim = data[data.iloc[:,0] == time].index[0]

        subset = data.iloc[lim:, : ].query("family == 9 & V == 2")
        if not subset.empty:
            first_enter = subset.iloc[0]
            buffer.append(first_enter)

    enter_zone2_df = pd.DataFrame(buffer, columns=data.columns)
    return enter_zone2_df

def get_drinking_times(data, index=2):
    #  Returns moments where the mouse is drinking, if index = 0 returns drinking moments when the trough is empty, index = 1 for a full trough
    if index == 2:
        drinking_times = data[(data.iloc[:,1] == 5) & (data.iloc[:, 4] == 1)].iloc[:, 0]
    else :
        drinking_times = data[(data.iloc[:,1] == 5) & (data.iloc[:,3] == index) & (data.iloc[:, 4] == 1)].iloc[:, 0]
    return drinking_times

def get_drinking_intervals(drinking_times, full_drinking_times, start_off_task):
    limit = 1000 #ms

    start_full_drinking_buffer = []
    end_full_drinking_buffer = []
    drinking_times = drinking_times.reset_index(drop=True)
    end = -1
    for i in range(len(full_drinking_times)):
        start = full_drinking_times.iloc[i]
        if start < end+1:
            continue
        next_drinking_times = drinking_times[drinking_times > start]
        previous_t = start
        for i in range(len(next_drinking_times)):
            next_t = next_drinking_times.iloc[i]
            if (next_t - previous_t) > limit:
                end = previous_t
                subset_off_task = start_off_task[(start_off_task>=start) & (start_off_task<=end)]
                if subset_off_task.empty:
                    start_full_drinking_buffer.append(start)
                    end_full_drinking_buffer.append(end)
                break
            else :
                previous_t = next_t

            if i == len(next_drinking_times)-1:
                end = next_drinking_times.iloc[-1]
                start_full_drinking_buffer.append(start)
                end_full_drinking_buffer.append(end)
                break
    start_full_drinking_df = pd.DataFrame(start_full_drinking_buffer, columns=["time"])
    end_full_drinking_df = pd.DataFrame(end_full_drinking_buffer, columns=["time"])

    mask = drinking_times.apply(lambda t: any((start_full_drinking_df.iloc[:,0] <= t) & (end_full_drinking_df.iloc[:,0] >= t)))

    empty_drinking_times = pd.DataFrame(drinking_times[~mask], columns=['time'] )
    empty_drinking_times['diff'] = empty_drinking_times.diff()
    empty_drinking_times['group'] = (empty_drinking_times['diff'] > limit).cumsum()
    empty_drinking_intervals = empty_drinking_times.groupby('group')['time'].agg(['first', 'last'])

    

    return start_full_drinking_df, end_full_drinking_df, empty_drinking_intervals

def in_off_task(zone1_times, zone2_times, drinking_times, pressing_times, max_t = 1200000):
    # Returns df containing start and end times of moments when the mouse is in or off task
    in_lever_task_start_buffer = []
    in_lever_task_end_buffer = []
    in_trough_task_start_buffer = []
    in_trough_task_end_buffer = []
    off_task_start_buffer = []
    off_task_end_buffer = []
    
    for start_z1_t, start_z2_t in itertools.zip_longest(zone1_times, zone2_times):
        if start_z1_t is not None:
            next_zone2_times = zone2_times[zone2_times > start_z1_t]
            next_zone1_times = zone1_times[zone1_times > start_z1_t]

            if not next_zone2_times.empty:
                next_zone2_t = next_zone2_times.iloc[0]
            else : 
                next_zone2_t = max_t

            if not next_zone1_times.empty:
                next_zone1_t = next_zone1_times.iloc[0]
            else :
                next_zone1_t = max_t

            end_t = min(next_zone1_t, next_zone2_t)

            subset_drinking = drinking_times[(drinking_times >= start_z1_t) & (drinking_times <= end_t)]
            subset_pressing = pressing_times[(pressing_times >= start_z1_t) & (pressing_times <= end_t)]

            if subset_drinking.empty & subset_pressing.empty:
                off_task_start_buffer.append(start_z1_t)
                off_task_end_buffer.append(end_t)
            else :
                in_trough_task_start_buffer.append(start_z1_t)
                in_trough_task_end_buffer.append(end_t)


        if start_z2_t is not None:
            next_zone1_times = zone1_times[zone1_times > start_z2_t]
            next_zone2_times = zone2_times[zone2_times > start_z2_t]

            if not next_zone1_times.empty:
                next_zone1_t = next_zone1_times.iloc[0]
            else :
                next_zone1_t = max_t

            if not next_zone2_times.empty:
                next_zone2_t = next_zone2_times.iloc[0]
            else : 
                next_zone2_t = max_t

            end_t = min(next_zone1_t, next_zone2_t)

            subset_drinking = drinking_times[(drinking_times >= start_z2_t) & (drinking_times <= end_t)]
            subset_pressing = pressing_times[(pressing_times >= start_z2_t) & (pressing_times <= end_t)]

            if subset_drinking.empty & subset_pressing.empty:
                off_task_start_buffer.append(start_z2_t)
                off_task_end_buffer.append(end_t)
            else :
                in_lever_task_start_buffer.append(start_z2_t)
                in_lever_task_end_buffer.append(end_t)
    
    in_lever_task_start_df = pd.DataFrame(in_lever_task_start_buffer, columns=["time"]).drop_duplicates()
    int_lever_task_end_df = pd.DataFrame(in_lever_task_end_buffer, columns=["time"]).drop_duplicates()
    in_trough_task_start_df = pd.DataFrame(in_trough_task_start_buffer, columns=["time"]).drop_duplicates()
    int_trough_task_end_df = pd.DataFrame(in_trough_task_end_buffer, columns=["time"]).drop_duplicates()
    off_task_start_df = pd.DataFrame(off_task_start_buffer, columns=["time"]).drop_duplicates()
    off_task_end_df = pd.DataFrame(off_task_end_buffer, columns=["time"]).drop_duplicates()
    
    return in_lever_task_start_df, int_lever_task_end_df, in_trough_task_start_df, int_trough_task_end_df, off_task_start_df, off_task_end_df

def merge_intervals(start_df, end_df):
    intervals = sorted(zip(start_df.iloc[:, 0], end_df.iloc[:, 0]))
    start_buffer = []
    end_buffer = []
    current_start, current_end = intervals[0]

    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            start_buffer.append(current_start)
            end_buffer.append(current_end)
            current_start, current_end = start, end
    start_buffer.append(current_start)
    end_buffer.append(current_end)
    return pd.DataFrame(start_buffer, columns=["time"]), pd.DataFrame(end_buffer, columns=["time"])


def plot_behaviors_levels(file_path):

    data = load_data(file_path)
    max_time = data.iloc[-1,0]
    fig = plt.figure(figsize=(12, 2))
    # zone_intervals = movie_data.main()

    # start_lever_zone = zone_intervals[zone_intervals['zone'] == 'lever_zone']["start_time"]
    # end_lever_zone = zone_intervals[zone_intervals['zone'] == 'lever_zone']["end_time"]
    # plt.barh(y=1.625, width=np.array(end_lever_zone) - np.array(start_lever_zone), left=start_lever_zone, height=0.25, color="yellow", edgecolor='black', label='lever zone')
    
    # start_trough_zone = zone_intervals[zone_intervals['zone'] == 'trough_zone']["start_time"]
    # end_trough_zone = zone_intervals[zone_intervals['zone'] == 'trough_zone']["end_time"]
    # plt.barh(y=1.625, width=np.array(end_trough_zone) - np.array(start_trough_zone), left=start_trough_zone, height=0.25, color="maroon", edgecolor='black', label='trough zone')

    # start_other_zone = zone_intervals[zone_intervals['zone'] == 'other']["start_time"]
    # end_other_zone = zone_intervals[zone_intervals['zone'] == 'other']["end_time"]
    # plt.barh(y=1.625, width=np.array(end_other_zone) - np.array(start_other_zone), left=start_other_zone, height=0.25, color="lavender", edgecolor='black', label='off zone')

    start_pressing_df, end_pressing_df = pressing_actions(data)
    start_pressing_times, end_pressing_times = start_pressing_df.iloc[:,0], end_pressing_df.iloc[:,0]
    plt.barh(y=1.25, width=np.array(end_pressing_times) - np.array(start_pressing_times), left=start_pressing_times, height=0.5, color="black", edgecolor='black', label='Press')

    enter_zone1_times = get_zones(data ,1)
    enter_zone2_times = get_zones(data, 2)
    
    drinking_times = get_drinking_times(data)
    full_drinking_times = get_drinking_times(data, 1)
    y_full_drinking = [1.875]*len(full_drinking_times)
    # average_gap = full_drinking_times.diff().mean()
    # print(average_gap)
    

    empty_drinking_times = get_drinking_times(data, 0)
    y_empty_drinking = [1.875]*len(empty_drinking_times)
    # average_gap = empty_drinking_times.diff().mean()
    # print(average_gap)
    

    start_in_lever_task, end_in_lever_task, start_in_trough_task, end_in_trough_task, start_off_task, end_off_task  = in_off_task(enter_zone1_times, enter_zone2_times, drinking_times, start_pressing_times, max_time)
    start_in_lever_task, end_in_lever_task = merge_intervals(start_in_lever_task, end_in_lever_task)
    start_in_trough_task, end_in_trough_task = merge_intervals(start_in_trough_task, end_in_trough_task)
    start_off_task, end_off_task = merge_intervals(start_off_task, end_off_task)
    plt.barh(y=.375, width=np.array(end_in_lever_task.iloc[:,0]) - np.array(start_in_lever_task.iloc[:,0]), left=start_in_lever_task.iloc[:,0], height=0.25, color="green", edgecolor='black', label='In lever task')
    plt.barh(y=.375, width=np.array(end_in_trough_task.iloc[:,0]) - np.array(start_in_trough_task.iloc[:,0]), left=start_in_trough_task.iloc[:,0], height=0.25, color="cornflowerblue", edgecolor='black', label='In trough task')
    plt.barh(y=0.125, width=np.array(end_off_task.iloc[:,0]) - np.array(start_off_task.iloc[:,0]), left=start_off_task.iloc[:,0], height=0.25, color="red", edgecolor='black', label='Off task')

    start_drinking_df, end_drinking_df, empty_drinking_intervals = get_drinking_intervals(drinking_times, full_drinking_times, start_off_task.iloc[:,0])
    start_drinking_times, end_drinking_times = start_drinking_df.iloc[:,0], end_drinking_df.iloc[:,0]
    plt.barh(y=1.875, width=np.array(end_drinking_times) - np.array(start_drinking_times), left=np.array(start_drinking_times), height=0.25, color="purple", edgecolor='black', label='Drinking full')
    
    start_empty_drinking_df, end_empty_drinking_df = empty_drinking_intervals.iloc[:,0], empty_drinking_intervals.iloc[:,1]
    plt.barh(y=1.875, width=np.array(end_empty_drinking_df) - np.array(start_empty_drinking_df), left=np.array(start_empty_drinking_df), height=0.25, color="orange", edgecolor='black', label='Drinking empty')
    plt.scatter(full_drinking_times,y_full_drinking, edgecolor='black',color='purple')
    plt.scatter(empty_drinking_times,y_empty_drinking, edgecolor='black',color='orange')

    seq_starts_df, seq_ends_df = action_sequences(data, start_pressing_df, end_pressing_df, drinking_times, pd.concat([start_in_lever_task, start_in_trough_task], ignore_index=True)["time"], pd.concat([end_in_lever_task, end_in_trough_task])["time"])
    seq_start_times, seq_end_times = seq_starts_df.iloc[:, 0], seq_ends_df.iloc[:,0]
    plt.barh(y=0.75, width=np.array(seq_end_times) - np.array(seq_start_times), left=seq_start_times, height=0.5, color="blue", edgecolor='black', label='Action sequence')

    for time in enter_zone1_times:
        plt.axvline(time, 0, 2, color="g", linestyle= '--')
    for time in enter_zone2_times:
        plt.axvline(time, 0, 2, color="r", linestyle= '--')
    plt.xlabel("Time (ms)")
    plt.legend(loc='upper right')
    plt.show()




dat_files = pd.read_csv(r".\behavioral_data\paths\paths_dat\M2_dat_exp10_to_16.csv")
for file in dat_files["File"]:
    print(file)
    plot_behaviors_levels(file)
# plot_behaviors_levels(dat_files["File"][2])
