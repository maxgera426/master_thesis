import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import itertools
import os
import re

column_names = ['time','family', 'num', 'P', 'V', 'L', 'R', 'T','W', 'X', 'Y', 'Z']


def load_data_from_dat(file_path):
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

def moving_in_zone2(start_in_task, end_in_task, start_action_sequence, end_action_sequence):
    start_lever_buffer = []
    end_lever_buffer = []
    start_z1_buffer = []
    end_z1_buffer = []
    for start, end in zip(start_in_task['time'], end_in_task['time']):
        next_action_start = start_action_sequence[(start_action_sequence >= start) & (start_action_sequence <= end)]
        next_action_end = end_action_sequence[(end_action_sequence >= start) & (end_action_sequence <= end)]
        if not next_action_start.empty:
            start_lever_buffer.append(start)
            end_lever_buffer.append(next_action_start.iloc[0])
        if not next_action_end.empty:
            start_z1_buffer.append(next_action_end.iloc[-1])
            end_z1_buffer.append(end)
    start_lever_df = pd.DataFrame(start_lever_buffer, columns=["time"])
    end_lever_df = pd.DataFrame(end_lever_buffer, columns=["time"])
    start_z1_df = pd.DataFrame(start_z1_buffer, columns=["time"])
    end_z1_df = pd.DataFrame(end_z1_buffer, columns=["time"])
    return start_lever_df, end_lever_df, start_z1_df, end_z1_df

def moving_in_zone1(start_in_task, end_in_task, drinking_times):
    start_trough_buffer = []
    end_trough_buffer = []
    start_z2_buffer = []
    end_z2_buffer = []
    for start, end in zip(start_in_task['time'], end_in_task['time']):
        next_drinking_times= drinking_times[(drinking_times >= start) & (drinking_times <= end)]
        if not next_drinking_times.empty:
            start_trough_buffer.append(start)
            end_trough_buffer.append(next_drinking_times.iloc[0])
            start_z2_buffer.append(next_drinking_times.iloc[-1])
            end_z2_buffer.append(end)
    start_trough_df = pd.DataFrame(start_trough_buffer, columns=['time'])
    end_trough_df = pd.DataFrame(end_trough_buffer, columns=['time'])
    start_z2_df = pd.DataFrame(start_z2_buffer, columns=['time'])
    end_z2_df = pd.DataFrame(end_z2_buffer, columns=['time'])
    return start_trough_df, end_trough_df, start_z2_df, end_z2_df

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
    if not intervals:
        return start_df, end_df
    
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

def true_drinking_times(start_full_drinking, end_full_drinking, drinking_times):
    mask = drinking_times.apply(lambda time: any(start <= time <= end for start, end in zip(start_full_drinking, end_full_drinking)))
    full_drinking_times = drinking_times[mask]
    empty_drinking_times = drinking_times[~mask]
    return full_drinking_times, empty_drinking_times

def compute_behavior_description(file_list, folder):
    for file_path in file_list:
        print("Processing file: ", os.path.basename(file_path))
        data = load_data_from_dat(file_path)
        max_time = data.iloc[-1,0]

        start_pressing_df, end_pressing_df = pressing_actions(data)
        start_pressing_times, end_pressing_times = start_pressing_df['time'], end_pressing_df['time']

        enter_zone1_times = get_zones(data ,1)
        enter_zone2_times = get_zones(data, 2)

        drinking_times = get_drinking_times(data)
        full_drinking_times = get_drinking_times(data, 1)
        empty_drinking_times = get_drinking_times(data, 0)

        start_in_lever_task, end_in_lever_task, start_in_trough_task, end_in_trough_task, start_off_task, end_off_task  = in_off_task(enter_zone1_times, enter_zone2_times,drinking_times, start_pressing_times, max_time)
        start_in_lever_task, end_in_lever_task = merge_intervals(start_in_lever_task, end_in_lever_task)
        start_in_trough_task, end_in_trough_task = merge_intervals(start_in_trough_task, end_in_trough_task)
        start_off_task, end_off_task = merge_intervals(start_off_task, end_off_task)

        start_drinking_df, end_drinking_df, empty_drinking_intervals = get_drinking_intervals(drinking_times, full_drinking_times, start_off_task.iloc[:,0])
        start_drinking_times, end_drinking_times = start_drinking_df.iloc[:,0], end_drinking_df.iloc[:,0]
        start_empty_drinking_times, end_empty_drinking_times = empty_drinking_intervals.iloc[:,0], empty_drinking_intervals.iloc[:,1]
        full_drinking_times, empty_drinking_times = true_drinking_times(start_drinking_times, end_drinking_times, drinking_times)

        seq_starts_df, seq_ends_df = action_sequences(data, start_pressing_df, end_pressing_df, drinking_times, pd.concat([start_in_lever_task, start_in_trough_task],ignore_index=True)   ["time"], pd.concat([end_in_lever_task, end_in_trough_task])["time"])
        seq_start_times, seq_end_times = seq_starts_df.iloc[:, 0], seq_ends_df.iloc[:,0]

        start_moving_to_lever, end_moving_to_lever, start_moving_to_z1, end_moving_to_z1 = moving_in_zone2(start_in_lever_task, end_in_lever_task, seq_start_times, seq_end_times)
        start_moving_to_trough, end_moving_to_trough, start_moving_to_z2, end_moving_to_z2 = moving_in_zone1(start_in_trough_task, end_in_trough_task, drinking_times)


        data_dict = {
            'Pressing Start': pd.Series(start_pressing_times.values),
            'Pressing End': pd.Series(end_pressing_times.values),
            'Enter Zone 1': pd.Series(enter_zone1_times.values),
            'Enter Zone 2': pd.Series(enter_zone2_times.values),
            'Licking full': pd.Series(full_drinking_times.values),
            'Licking Empty': pd.Series(empty_drinking_times.values),
            'Drinking Full Start': pd.Series(start_drinking_times.values),
            'Drinking Full End': pd.Series(end_drinking_times.values),
            'Drinking Empty Start': pd.Series(start_empty_drinking_times.values),
            'Drinking Empty End': pd.Series(end_empty_drinking_times.values),
            'In Lever Task Start': pd.Series(start_in_lever_task['time'].values),
            'In Lever Task End': pd.Series(end_in_lever_task['time'].values),
            'In Trough Task Start': pd.Series(start_in_trough_task['time'].values),
            'In Trough Task End': pd.Series(end_in_trough_task['time'].values),
            'Off Task Start': pd.Series(start_off_task.iloc[:,0].values),
            'Off Task End': pd.Series(end_off_task.iloc[:,0].values),
            'Sequence Start': pd.Series(seq_start_times.values),
            'Sequence End': pd.Series(seq_end_times.values),
            'Moving To Lever Start' : pd.Series(start_moving_to_lever['time'].values),
            'Moving To Lever End' :  pd.Series(end_moving_to_lever['time'].values),
            'Moving To Trough Start' : pd.Series(start_moving_to_trough['time'].values),
            'Moving To Trough End' :  pd.Series(end_moving_to_trough['time'].values),
            'Moving To Zone 1 Start' : pd.Series(start_moving_to_z1['time'].values),
            'Moving To Zone 1 End' :  pd.Series(end_moving_to_z1['time'].values),
            'Moving To Zone 2 Start' : pd.Series(start_moving_to_z2['time'].values),
            'Moving To Zone 2 End' :  pd.Series(end_moving_to_z2['time'].values),
        }
        exp_folder = os.path.dirname(file_path)
        behaviors_df = pd.DataFrame(data_dict, dtype=object)
        behaviors_df.to_csv(folder + '/' + os.path.basename(os.path.dirname(exp_folder))+ '_' + os.path.basename(exp_folder) + '_behavior_description.csv', index=False)

    return

def compute_partial_behavior_description(file_list, folder, n):
    for file_path in file_list:
        print("Processing file: ", os.path.basename(file_path))
        data = load_data_from_behavior_csv(file_path)
        exp_length = 1200000 # miliseconds
        parts = [(i*exp_length/n, exp_length/n*(i+1)) for i in range(n)]
        for start, end in parts:
            column_names = ['Pressing Start', 'Pressing End', 'Enter Zone 1', 'Enter Zone 2', 'Licking full', 'Licking Empty', 'Drinking Full Start', 'Drinking Full End', 'Drinking Empty Start', 'Drinking Empty End', 'In Lever Task Start', 'In Lever Task End', 'In Trough Task Start', 'In Trough Task End', 'Off Task Start', 'Off Task End', 'Sequence Start', 'Sequence End']

            partial_pressing_start = data['Pressing Start'][(data['Pressing Start'] >= start) & (data['Pressing Start'] <= end)]
            partial_pressing_end = data['Pressing End'][(data['Pressing End'] >= start) & (data['Pressing End'] <= end)]
            if len(partial_pressing_start) != len(partial_pressing_end):
                partial_pressing_start, partial_pressing_end = correct_cut(start, end, partial_pressing_start, partial_pressing_end)
            
            partial_licking_full = data['Licking full'][(data['Licking full'] >= start) & (data['Licking full'] <= end)]
            partial_licking_empty = data['Licking Empty'][(data['Licking Empty'] >= start) & (data['Licking Empty'] <= end)]

            partial_enter_z1 = data['Enter Zone 1'][(data['Enter Zone 1'] >= start) & (data['Enter Zone 1'] <= end)]
            partial_enter_z2 = data['Enter Zone 2'][(data['Enter Zone 2'] >= start) & (data['Enter Zone 2'] <= end)]

            
            partial_drinking_full_start = data['Drinking Full Start'][(data['Drinking Full Start'] >= start) & (data['Drinking Full Start'] <= end)]
            partial_drinking_full_end = data['Drinking Full End'][(data['Drinking Full End'] >= start) & (data['Drinking Full End'] <= end)]
            if len(partial_drinking_full_start) != len(partial_drinking_full_end):
                partial_drinking_full_start, partial_drinking_full_end = correct_cut(start, end, partial_drinking_full_start, partial_drinking_full_end)

            partial_drinking_empty_start = data['Drinking Empty Start'][(data['Drinking Empty Start'] >= start) & (data['Drinking Empty Start'] <= end)]
            partial_drinking_empty_end = data['Drinking Empty End'][(data['Drinking Empty End'] >= start) & (data['Drinking Empty End'] <= end)]
            if len(partial_drinking_empty_start) != len(partial_drinking_empty_end):
                partial_drinking_empty_start, partial_drinking_empty_end = correct_cut(start, end, partial_drinking_empty_start, partial_drinking_empty_end)

            partial_lever_start = data['In Lever Task Start'][(data['In Lever Task Start'] >= start) & (data['In Lever Task Start'] <= end)]
            partial_lever_end = data['In Lever Task End'][(data['In Lever Task End'] >= start) & (data['In Lever Task End'] <= end)]
            if len(partial_lever_start) != len(partial_lever_end):
                partial_lever_start, partial_lever_end = correct_cut(start, end, partial_lever_start, partial_lever_end)

            partial_trough_start = data['In Trough Task Start'][(data['In Trough Task Start'] >= start) & (data['In Trough Task Start'] <= end)]
            partial_trough_end = data['In Trough Task End'][(data['In Trough Task End'] >= start) & (data['In Trough Task End'] <= end)]
            if len(partial_trough_start) != len(partial_trough_end):
                partial_trough_start, partial_trough_end = correct_cut(start, end, partial_trough_start, partial_trough_end)

            partial_off_start = data['Off Task Start'][(data['Off Task Start'] >= start) & (data['Off Task Start'] <= end)]
            partial_off_end = data['Off Task End'][(data['Off Task End'] >= start) & (data['Off Task End'] <= end)]
            if len(partial_off_start) != len(partial_off_end):
                partial_off_start, partial_off_end = correct_cut(start, end, partial_off_start, partial_off_end)

            partial_sequence_start = data['Sequence Start'][(data['Sequence Start'] >= start) & (data['Sequence Start'] <= end)]
            partial_sequence_end = data['Sequence End'][(data['Sequence End'] >= start) & (data['Sequence End'] <= end)]
            if len(partial_sequence_start) != len(partial_sequence_end):
                partial_sequence_start, partial_sequence_end = correct_cut(start, end, partial_sequence_start, partial_sequence_end)

            partial_moving_lever_start = data['Moving To Lever Start'][(data['Moving To Lever Start'] >= start) & (data['Moving To Lever Start'] <= end)]
            partial_moving_lever_end = data['Moving To Lever End'][(data['Moving To Lever End'] >= start) & (data['Moving To Lever End'] <= end)]
            if len(partial_moving_lever_start) != len(partial_moving_lever_end):
                partial_moving_lever_start, partial_moving_lever_end = correct_cut(start, end, partial_moving_lever_start, partial_moving_lever_end)
            
            partial_moving_trough_start = data['Moving To Trough Start'][(data['Moving To Trough Start'] >= start) & (data['Moving To Trough Start'] <= end)]
            partial_moving_trough_end = data['Moving To Trough End'][(data['Moving To Trough End'] >= start) & (data['Moving To Trough End'] <= end)]
            if len(partial_moving_trough_start) != len(partial_moving_trough_end):
                partial_moving_trough_start, partial_moving_trough_end = correct_cut(start, end, partial_moving_trough_start, partial_moving_trough_end)

            partial_moving_z1_start = data['Moving To Zone 1 Start'][(data['Moving To Zone 1 Start'] >= start) & (data['Moving To Zone 1 Start'] <= end)]
            partial_moving_z1_end = data['Moving To Zone 1 End'][(data['Moving To Zone 1 End'] >= start) & (data['Moving To Zone 1 End'] <= end)]
            if len(partial_moving_z1_start) != len(partial_moving_z1_end):
                partial_moving_z1_start, partial_moving_z1_end = correct_cut(start, end, partial_moving_z1_start, partial_moving_z1_end)
            
            partial_moving_z2_start = data['Moving To Zone 2 Start'][(data['Moving To Zone 2 Start'] >= start) & (data['Moving To Zone 2 Start'] <= end)]
            partial_moving_z2_end = data['Moving To Zone 2 End'][(data['Moving To Zone 2 End'] >= start) & (data['Moving To Zone 2 End'] <= end)]
            if len(partial_moving_z2_start) != len(partial_moving_z2_end):
                partial_moving_z2_start, partial_moving_z2_end = correct_cut(start, end, partial_moving_z2_start, partial_moving_z2_end)

            columns = [partial_pressing_start.reset_index(drop=True),
            partial_pressing_end.reset_index(drop=True),
            partial_enter_z1.reset_index(drop=True),
            partial_enter_z2.reset_index(drop=True),
            partial_licking_full.reset_index(drop=True),
            partial_licking_empty.reset_index(drop=True),
            partial_drinking_full_start.reset_index(drop=True),
            partial_drinking_full_end.reset_index(drop=True),
            partial_drinking_empty_start.reset_index(drop=True),
            partial_drinking_empty_end.reset_index(drop=True),
            partial_lever_start.reset_index(drop=True),
            partial_lever_end.reset_index(drop=True),
            partial_trough_start.reset_index(drop=True),
            partial_trough_end.reset_index(drop=True),
            partial_off_start.reset_index(drop=True),
            partial_off_end.reset_index(drop=True),
            partial_sequence_start.reset_index(drop=True),
            partial_sequence_end.reset_index(drop=True),
            partial_moving_lever_start.reset_index(drop=True),
            partial_moving_lever_end.reset_index(drop=True),
            partial_moving_trough_start.reset_index(drop=True),
            partial_moving_trough_end.reset_index(drop=True),
            partial_moving_z1_start.reset_index(drop=True),
            partial_moving_z1_end.reset_index(drop=True),
            partial_moving_z2_start.reset_index(drop=True),
            partial_moving_z2_end.reset_index(drop=True)]

            exp_name = re.search("Exp ...", os.path.basename(file_path)).group()
            mouse_name = re.search("M.", os.path.basename(file_path)).group()
            save_folder = folder + '/' + exp_name + '/' + f"{n}_subdivisions"
            os.makedirs(save_folder, exist_ok=True)
            behaviors_df = pd.concat(columns, axis=1)
            behaviors_df.to_csv(save_folder + '/' + mouse_name + '_' + exp_name + "_" + f"{start}_{end}" + '_behavior_description.csv', index=False)
    return 0

def correct_cut(start, end, start_times, end_times):
    if start_times.iloc[0] >= end_times.iloc[0]:
        new_series = pd.Series([start], name=start_times.name)
        start_times = pd.concat([start_times, new_series]).sort_values().reset_index(drop=True)
    if end_times.iloc[-1] <= start_times.iloc[-1]:
        new_series = pd.Series([end], name=end_times.name)
        end_times = pd.concat([end_times, new_series]).sort_values().reset_index(drop=True)
    return start_times, end_times

def plot_single_experiment(file_path):
    data = load_data_from_behavior_csv(file_path)
    title = re.search("Exp ...", os.path.basename(file_path)).group()
    fig = plt.figure(figsize=(12, 2))

    start_pressing_times = data['Pressing Start'].dropna()
    end_pressing_times = data['Pressing End'].dropna()
    full_drinking_times = data['Licking full'].dropna()
    empty_drinking_times = data['Licking Empty'].dropna()
    start_in_lever_task = data['In Lever Task Start'].dropna()
    end_in_lever_task = data['In Lever Task End'].dropna()
    start_in_trough_task = data['In Trough Task Start'].dropna()
    end_in_trough_task = data['In Trough Task End'].dropna()
    start_off_task = data['Off Task Start'].dropna()
    end_off_task = data['Off Task End'].dropna()
    start_drinking_times = data['Drinking Full Start']
    end_drinking_times = data['Drinking Full End']
    start_empty_drinking_df = data['Drinking Empty Start'].dropna()
    end_empty_drinking_df = data['Drinking Empty End'].dropna()
    seq_start_times = data['Sequence Start'].dropna()
    seq_end_times = data['Sequence End'].dropna()
    start_moving_to_lever = data["Moving To Lever Start"].dropna()
    end_moving_to_lever = data["Moving To Lever End"].dropna()
    start_moving_to_z1 = data["Moving To Zone 1 Start"].dropna()
    end_moving_to_z1 = data['Moving To Zone 1 End'].dropna()
    start_moving_to_trough = data["Moving To Trough Start"].dropna()
    end_moving_to_trough = data["Moving To Trough End"].dropna()
    start_moving_to_z2 = data["Moving To Zone 2 Start"].dropna()
    end_moving_to_z2 = data["Moving To Zone 2 End"].dropna()

    plt.barh(y=1.25, width=np.array(end_pressing_times) - np.array(start_pressing_times), left=start_pressing_times, height=0.5, color="black", edgecolor='black', label='Press')
    plt.barh(y=.125, width=np.array(end_in_lever_task) - np.array(start_in_lever_task), left=start_in_lever_task, height=0.25, color="red", edgecolor='black', label='In lever task')
    plt.barh(y=.125, width=np.array(end_in_trough_task) - np.array(start_in_trough_task), left=start_in_trough_task, height=0.25, color="blue", edgecolor='black', label='In trough task')
    plt.barh(y=0.125, width=np.array(end_off_task) - np.array(start_off_task), left=start_off_task, height=0.25, color="gray", edgecolor='black', label='Off task')
    plt.barh(y=1.875, width=np.array(end_drinking_times) - np.array(start_drinking_times), left=np.array(start_drinking_times), height=0.25, color="purple", edgecolor='black', label='Drinking full')
    plt.barh(y=1.875, width=np.array(end_empty_drinking_df) - np.array(start_empty_drinking_df), left=np.array(start_empty_drinking_df), height=0.25, color="orange", edgecolor='black', label='Drinking empty')
    y_empty_drinking = [1.875]*len(empty_drinking_times)
    y_full_drinking = [1.875]*len(full_drinking_times)
    plt.scatter(full_drinking_times,y_full_drinking, edgecolor='black',color='purple')
    plt.scatter(empty_drinking_times,y_empty_drinking, edgecolor='black',color='orange')
    plt.barh(y=0.75, width=np.array(seq_end_times) - np.array(seq_start_times), left=seq_start_times, height=0.5, color="green", edgecolor='black', label='Action sequence')
    plt.barh(y=.375, width = end_moving_to_lever - start_moving_to_lever, left= start_moving_to_lever, height=0.25, color="darkred", edgecolor='black', label="moving to lever")
    plt.barh(y=.375, width = end_moving_to_z1 - start_moving_to_z1, left= start_moving_to_z1, height=0.25, color="tomato", edgecolor='black', label="moving to z1")
    plt.barh(y=.375, width = end_moving_to_trough - start_moving_to_trough, left= start_moving_to_trough, height=0.25, color="navy", edgecolor='black', label="moving to lever")
    plt.barh(y=.375, width = end_moving_to_z2 - start_moving_to_z2, left= start_moving_to_z2, height=0.25, color="mediumslateblue", edgecolor='black', label="moving to z2")
    plt.xlabel("Time (ms)")
    fig.legend(loc='center right')
    plt.title(title, loc='left', fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

def plot_behavior_levels(file_list):
    n = len(file_list)
    if n == 0:
        print("please give a file path or a list of file paths")
    elif n == 1 : 
        plot_single_experiment(file_list[0])
    else :
        fig, axes = plt.subplots(n, figsize=(12, 2*n))
        for ax, file_path in zip(axes, file_list):
            data = load_data_from_behavior_csv(file_path)
            print("Plotting file : ", file_path)

            start_pressing_times = data['Pressing Start'].dropna()
            end_pressing_times = data['Pressing End'].dropna()
            full_drinking_times = data['Licking full'].dropna()
            empty_drinking_times = data['Licking Empty'].dropna()
            start_in_lever_task = data['In Lever Task Start'].dropna()
            end_in_lever_task = data['In Lever Task End'].dropna()
            start_in_trough_task = data['In Trough Task Start'].dropna()
            end_in_trough_task = data['In Trough Task End'].dropna()
            start_off_task = data['Off Task Start'].dropna()
            end_off_task = data['Off Task End'].dropna()
            start_drinking_times = data['Drinking Full Start']
            end_drinking_times = data['Drinking Full End']
            start_empty_drinking_df = data['Drinking Empty Start'].dropna()
            end_empty_drinking_df = data['Drinking Empty End'].dropna()
            seq_start_times = data['Sequence Start'].dropna()
            seq_end_times = data['Sequence End'].dropna()
            start_moving_to_lever = data["Moving To Lever Start"].dropna()
            end_moving_to_lever = data["Moving To Lever End"].dropna()
            start_moving_to_z1 = data["Moving To Zone 1 Start"].dropna()
            end_moving_to_z1 = data['Moving To Zone 1 End'].dropna()
            start_moving_to_trough = data["Moving To Trough Start"].dropna()
            end_moving_to_trough = data["Moving To Trough End"].dropna()
            start_moving_to_z2 = data["Moving To Zone 2 Start"].dropna()
            end_moving_to_z2 = data["Moving To Zone 2 End"].dropna()

            ax.barh(y=1.25, width=np.array(end_pressing_times) - np.array(start_pressing_times), left=start_pressing_times, height=0.5, color="black", edgecolor='black', label='Press')
            ax.barh(y=.125, width=np.array(end_in_lever_task) - np.array(start_in_lever_task), left=start_in_lever_task, height=0.25, color="red", edgecolor='black', label='In lever task')
            ax.barh(y=.125, width=np.array(end_in_trough_task) - np.array(start_in_trough_task), left=start_in_trough_task, height=0.25, color="blue", edgecolor='black', label='In trough task')
            ax.barh(y=0.125, width=np.array(end_off_task) - np.array(start_off_task), left=start_off_task, height=0.25, color="gray", edgecolor='black', label='Off task')
            ax.barh(y=1.875, width=np.array(end_drinking_times) - np.array(start_drinking_times), left=np.array(start_drinking_times), height=0.25, color="purple", edgecolor='black', label='Drinking full')
            ax.barh(y=1.875, width=np.array(end_empty_drinking_df) - np.array(start_empty_drinking_df), left=np.array(start_empty_drinking_df), height=0.25, color="orange", edgecolor='black', label='Drinking empty')
            y_empty_drinking = [1.875]*len(empty_drinking_times)
            y_full_drinking = [1.875]*len(full_drinking_times)
            ax.scatter(full_drinking_times,y_full_drinking, edgecolor='black',color='purple')
            ax.scatter(empty_drinking_times,y_empty_drinking, edgecolor='black',color='orange')
            ax.barh(y=0.75, width=np.array(seq_end_times) - np.array(seq_start_times), left=seq_start_times, height=0.5, color="green", edgecolor='black', label='Action sequence')
            ax.barh(y=.375, width = end_moving_to_lever - start_moving_to_lever, left= start_moving_to_lever, height=0.25, color="darkred", edgecolor='black', label="moving to lever")
            ax.barh(y=.375, width = end_moving_to_z1 - start_moving_to_z1, left= start_moving_to_z1, height=0.25, color="tomato", edgecolor='black', label="moving to z1")
            ax.barh(y=.375, width = end_moving_to_trough - start_moving_to_trough, left= start_moving_to_trough, height=0.25, color="navy", edgecolor='black', label="moving to lever")
            ax.barh(y=.375, width = end_moving_to_z2 - start_moving_to_z2, left= start_moving_to_z2, height=0.25, color="mediumslateblue", edgecolor='black', label="moving to z2")

            ax.set_xlabel("Time (ms)")
            ax.set_title(re.search("Exp ...", os.path.basename(file_path)).group(), loc='left', fontweight= 'bold')

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        title = "Behavior levels"
        fig.suptitle(title, fontweight='bold') 
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
    return


## Metrics computation on behavior description

def load_data_from_behavior_csv(file_path):
    # Loads the data from behavior description csv. This csv is saved using the above save_behavior_description(args) function
    data = pd.read_csv(file_path)
    return data

def compute_metrics(file_list):
    column_names = ['Pressing Start', 'Pressing End', 'Enter Zone 1', 'Enter Zone 2', 'Licking full', 'Licking Empty', 'Drinking Full Start', 'Drinking Full End', 'Drinking Empty Start', 'Drinking Empty End', 'In Lever Task Start', 'In Lever Task End', 'In Trough Task Start', 'In Trough Task End', 'Off Task Start', 'Off Task End', 'Sequence Start', 'Sequence End']
    experiment_list = []
    list_press_counts= []
    experiments_stats = {}
    for file_path in file_list:
        print("\nFile: ", os.path.basename(file_path))
        exp_name = re.search("Exp ...", os.path.basename(file_path)).group()
        experiment_list.append(exp_name)
        experiments_stats[exp_name] = {}
        data = load_data_from_behavior_csv(file_path)
        presses_statistics = get_press_seq_statistics(data["Pressing Start"].dropna(), data["Sequence Start"].dropna(), data["Sequence End"].dropna())
        drinking_statistics = get_drinking_statistics(data['Drinking Full Start'].dropna(), data['Drinking Full End'].dropna(), data['Licking full'], data['Drinking Empty Start'].dropna(), data['Drinking Empty End'].dropna(), data['Licking Empty'])
        task_statistics = get_task_statistics(data['In Lever Task Start'].dropna(), data['In Lever Task End'].dropna(), data['In Trough Task Start'].dropna(), data['In Trough Task End'].dropna(), data['Off Task Start'].dropna(), data['Off Task End'].dropna())
        experiments_stats[exp_name]["full session"] = [presses_statistics, drinking_statistics, task_statistics]
    return experiments_stats

def get_press_seq_statistics(presses_start, seq_start, seq_end):
    list_n_presses = []
    if not seq_start.empty:
        seq_end = seq_end[seq_end>= seq_start.iloc[0]]
    else:
        print("No action sequence in this time interval")
        list_n_presses.append(0)
    for start, end in zip(seq_start, seq_end):
        presses = presses_start[(presses_start >= start) & (presses_start <= end)]
        n_presses = len(presses)
        list_n_presses.append(n_presses)
    # list_n_presses.sort()
    statistics_dict = {
        'median' : np.median(list_n_presses),
        'mean' : np.mean(list_n_presses),
        'total_presses' : np.sum(list_n_presses),
        'total_seq' : len(list_n_presses),
        'max_presses' : np.max(list_n_presses),
        'min_presses' : np.min(list_n_presses),
        'std' : np.std(list_n_presses),
        'list_n_presses' : list_n_presses
    }
    return statistics_dict

def get_drinking_statistics(drinking_full_start, drinking_full_end, licking_full, drinking_empty_start, drinking_empty_end, licking_empty):
    if not drinking_full_start.any():
        print("No drinking full behavior in this time interval")
        return
    
    n_full_visits = len(drinking_full_start)
    n_empty_visits = len(drinking_empty_start)
    diff_full = drinking_full_end - drinking_full_start
    diff_empty = drinking_empty_end - drinking_empty_start
    total_time_full_drinking = np.sum(diff_full)/1000
    total_time_empty_drinking = np.sum(diff_empty)/1000
    total_time_drinking = total_time_full_drinking + total_time_empty_drinking

    list_n_licking_full = []
    for start, end in zip(drinking_full_start, drinking_full_end):
        licks = licking_full[(licking_full >= start) & (licking_full <= end)]
        n_licks = len(licks)
        list_n_licking_full.append(n_licks)
    
    list_n_licking_empty = []
    for start, end in zip(drinking_empty_start, drinking_empty_end):
        licks = licking_empty[(licking_empty >= start) & (licking_empty <= end)]
        n_licks = len(licks)
        list_n_licking_empty.append(n_licks)


    statistics_dict = {
        'n_full_visits' : n_full_visits,
        'n_empty_visits' : n_empty_visits,
        'total_time_full_drinking' : total_time_full_drinking/total_time_drinking,
        'total_time_empty_drinking' : total_time_empty_drinking/total_time_drinking,
        'mean_full_drinking' : np.mean(diff_full)/1000,
        'mean_empty_drinking' : np.mean(diff_empty)/1000,
        'list_n_lickings_full' : list_n_licking_full,
        'max_full_licking' : np.max(list_n_licking_full),
        'min_full_licking' : np.min(list_n_licking_full),
        'mean_full_licking' : np.mean(list_n_licking_full),
        'median_full_licking' : np.median(list_n_licking_full),  
        'list_n_lickings_empty' : list_n_licking_empty,
        'max_empty_licking' : np.max(list_n_licking_empty),
        'min_empty_licking' : np.min(list_n_licking_empty),
        'mean_empty_licking' : np.mean(list_n_licking_empty),
        'median_empty_licking' : np.median(list_n_licking_empty),  
    }
    return statistics_dict

def get_task_statistics(lever_start, lever_end, trough_start, trough_end, off_start, off_end):
    diff_lever = lever_end - lever_start
    diff_trough = trough_end - trough_start
    diff_off = off_end - off_start

    statistics_dict = {
        'total_time_lever_task' : np.sum(diff_lever)/1000,
        'total_time_trough_task' : np.sum(diff_trough)/1000,
        'total_time_off_task' : np.sum(diff_off)/1000
    }
    return statistics_dict

def get_total_time(start, end):
    diff = end - start
    total_time = np.sum(diff)/1000
    return total_time

def plot_statistics(statistics, stat_types, partial = False, n = 0):
    experiment_list = statistics.keys()
    for type in stat_types:
        if type == "Press":
            list_press_counts = []
            for exp in experiment_list:
                list_press_counts.append(statistics[exp]["full session"][0]["list_n_presses"])
            plt.figure()
            bp = plt.boxplot(list_press_counts, tick_labels=experiment_list, showmeans=True)
            plt.title("Presses per Action Sequence box plots", fontweight = "bold")
            plt.ylabel("Number of presses")
            plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'])
            plt.show()

            if partial :
                list_press_counts_partial = []
                for exp in experiment_list:
                    list_press_counts_exp = []
                    for i in range(1, n+1):
                        list_press_counts_exp.append(statistics[exp]["partial sessions"][f"Part_{i}"][0]["list_n_presses"])
                    list_press_counts_partial.append(list_press_counts_exp)
                x = np.arange(len(experiment_list))*4
                width = 0.6
                plt.figure()
                i = 0
                for exp in list_press_counts_partial:
                    positions = [x[i]-1.2, x[i]-0.4, x[i] + 0.4, x[i] + 1.2]
                    bp = plt.boxplot(exp, positions=positions, widths=width)
                    i+=1
                plt.xticks(x, experiment_list, rotation=45, ha="right")
                plt.show()


        elif type == "Drinking":
            full_drinking_times = []
            list_n_full_visits = []
            empty_drinking_times = []
            list_n_empty_visits = []
            for exp in experiment_list:
                full_drinking_times.append(statistics[exp]["full session"][1]["total_time_full_drinking"])
                empty_drinking_times.append(statistics[exp]["full session"][1]["total_time_empty_drinking"])
                list_n_full_visits.append(statistics[exp]["full session"][1]["n_full_visits"])
                list_n_empty_visits.append(statistics[exp]["full session"][1]["n_empty_visits"])

            x = np.arange(len(experiment_list))
            width = 0.25
            # add plot of n visits
            fig, ax1 = plt.subplots(figsize=(10, 5))  # Create the figure and primary axis

            # Primary axis (Drinking times)
            ax1.bar(x - width/2, full_drinking_times, width, label="Drinking Full", color='purple')
            ax1.bar(x + width/2, empty_drinking_times, width, label="Drinking Empty", color='orange')
            ax1.set_ylabel("Total Drinking Time (%)", color='black')
            ax1.tick_params(axis='y', labelcolor='black')

            # Secondary axis (Number of visits)
            ax2 = ax1.twinx()  # Create a second y-axis
            ax2.plot(x, list_n_full_visits, "-o", label="Number of Full Visits", color='blue')
            ax2.plot(x, list_n_empty_visits, "-x", label="Number of Empty Visits", color='red')
            ax2.set_ylabel("Number of Visits", color='black')
            ax2.tick_params(axis='y', labelcolor='black')
            plt.xticks(x, experiment_list, rotation=45, ha="right")
            ax1.set_title("Drinking Times and Number of Visits for Each Experiment")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")

            plt.tight_layout()
            plt.show()

        elif type == "Task":
            lever_task_times = []
            trough_task_times = []
            off_task_times = []
            for exp in experiment_list:
                lever_task_times.append(statistics[exp]["full session"][2]["total_time_lever_task"])
                trough_task_times.append(statistics[exp]["full session"][2]["total_time_trough_task"])
                off_task_times.append(statistics[exp]["full session"][2]["total_time_off_task"])
            
            x = np.arange(len(experiment_list))
            width = 0.25

            plt.figure()
            plt.bar(x - width, lever_task_times, width, label= "In lever task")
            plt.bar(x, trough_task_times, width, label="In trough task")
            plt.bar(x + width, off_task_times, width, label="Off task")
            plt.ylabel("Time (s)")
            plt.title("Task times for each experiments")
            plt.xticks(x, experiment_list, rotation=45, ha="right")
            plt.legend()
            plt.show()

    return 0

def main():
    ##COMPUTE BEHAVIOR LEVELS DURING EXPERIMENTS
    # file_list = pd.read_csv(r".\behavioral_data\paths\paths_dat\M15_dat_exp10_to_17.csv")["File"]
    # compute_behavior_description(file_list, r".\behavioral_data\behavior descriptions\full session")
    ##COMPUTE METRICS ON SAVED BEHAVIOR DESCRIPTIONS
    # file_list = [r'.\behavioral_data\behavior descriptions\full session\M2\\' + f for f in os.listdir(r'.\behavioral_data\behavior descriptions\full session\M2\\')]
    # compute_partial_behavior_description([r"C:\Users\maxge\OneDrive - Université Libre de Bruxelles\MA2\Mémoire\master_thesis\behavioral_data\behavior descriptions\full session\M2 - Jun24_Exp 010_behavior_description.csv"], r'C:\Users\maxge\OneDrive - Université Libre de Bruxelles\MA2\Mémoire\master_thesis\behavioral_data\behavior descriptions\partial sessions', 4)
    # statistics = compute_metrics(file_list)
    # plot_statistics(statistics, ["Press", "Drinking", "Task"], partial=False, n=4)
    
    plot_behavior_levels([r"behavioral_data\behavior descriptions\full session\M2\M2 - Jun24_Exp 010_behavior_description.csv"])
    # plot_behavior_levels([r"C:\Users\maxge\OneDrive - Université Libre de Bruxelles\MA2\Mémoire\master_thesis\behavioral_data\behavior descriptions\full session\M2 - Jun24_Exp 010_behavior_description.csv"])
if __name__ == "__main__": 
    main()