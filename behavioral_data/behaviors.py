import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import seaborn as sns
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
                start = single_press_action[(single_press_action.iloc[:, 4] == 1)].iloc[0,:]
                end = single_press_action[(single_press_action.iloc[:, 4] == 0)].iloc[0,:]

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

def action_sequences(data, start_pressing, end_pressing, enter_z1, enter_z2):

    start_buffer = []
    end_buffer = []
    max_t = np.max(data.iloc[:,0])

    for start in enter_z2:
        next_z1 = enter_z1[enter_z1 > start]
        next_z2 = enter_z2[enter_z2 > start]

        if next_z1.any() and next_z2.any():
            end = min(next_z1.iloc[0], next_z2.iloc[0])
        elif next_z1.any():
            end = next_z1.iloc[0]
        elif next_z2.any():
            end = next_z2.iloc[0]
        else :
            end = max_t
        
        subset_start = start_pressing[(start_pressing.iloc[:,0] >= start) & (start_pressing.iloc[:,0] <= end)].reset_index(drop=True)

        if len(subset_start) != 0 :
            seq_start = subset_start.iloc[0, :]
            last_press = subset_start.iloc[-1, 0]

            subset_end = end_pressing[(end_pressing.iloc[:,0] >= last_press)].reset_index(drop=True)
            seq_end = subset_end.iloc[0, :]

            start_buffer.append(tuple(seq_start))
            end_buffer.append(tuple(seq_end))
                
    seq_starts_df = pd.DataFrame(start_buffer, columns= data.columns).drop_duplicates().reset_index(drop=True)
    seq_ends_df = pd.DataFrame(end_buffer, columns= data.columns).drop_duplicates().reset_index(drop=True)

    return seq_starts_df, seq_ends_df

def off_task(enter_z1, enter_z2, pressing_times, drinking_times, max_t):

    all_zones = pd.concat([pd.Series([0]), enter_z1, enter_z2, pd.Series([max_t])]).sort_values()
    intervals = []

    last_end = None
    
    for start in all_zones:
        if last_end is not None and start <= last_end:
            continue

        next_zone_times = all_zones[all_zones > start]


        if not next_zone_times.empty:
            sorted_times = next_zone_times.sort_values(ascending=False)
            for end in sorted_times:
                subset_drinking = drinking_times[(drinking_times >= start) & (drinking_times <= end)]
                subset_pressing = pressing_times[(pressing_times >= start) & (pressing_times <= end)]

                if subset_drinking.empty & subset_pressing.empty:
                    intervals.append([start, end])
                    last_end = end
                    break

    off_task_intervals = pd.DataFrame(intervals, columns=["start", "end"])

    return off_task_intervals

def moving_to_lever_zone(start_seq, end_drinking, off_task_start, off_task_end, max_t):
    intervals = []

    for start in end_drinking:
        subset_end = start_seq[start_seq>start]

        if not subset_end.empty:
            end = subset_end.iloc[0]
        else : 
            end = max_t

        subset_off_task_start = off_task_start[(off_task_start >= start) & (off_task_start<=end)]
        subset_drinking = end_drinking[(end_drinking > start) & (end_drinking <= end)]

        if subset_off_task_start.empty & subset_drinking.empty:
            intervals.append([start, end])
        
        elif (subset_off_task_start.empty) & (not subset_drinking.empty):
            continue

        elif (not subset_off_task_start.empty) & (not subset_drinking.empty):
            off_task_stop = subset_off_task_start.iloc[0]
            drinking_stop = subset_drinking.iloc[0]

            if drinking_stop < off_task_stop: continue
            else : 
                intervals.append([start, off_task_stop])
        else:
            stop = subset_off_task_start.iloc[0]
            start2 = off_task_end[(off_task_end >= start) & (off_task_end<=end)].iloc[-1]
            intervals.append([start, stop])
            intervals.append([start2, end])

    for start in off_task_end:
        subset_end = start_seq[start_seq>start]

        if not subset_end.empty:
            end = subset_end.iloc[0]

            subset_off_task_start = off_task_start[(off_task_start >= start) & (off_task_start<=end)]
            subset_drinking = end_drinking[(end_drinking > start) & (end_drinking <= end)]

            if subset_off_task_start.empty & subset_drinking.empty:
                intervals.append([start, end])
            else : 
                continue

    
    moving_to_lever_zone_intervals = pd.DataFrame(intervals, columns=["start", "end"]).sort_values(by="start")
    return moving_to_lever_zone_intervals

def moving_to_trough_zone(end_seq, start_drinking, off_task_start, off_task_end, max_t):
    intervals = []

    for start in end_seq:
        subset_end = start_drinking[start_drinking>start]

        if not subset_end.empty:
            end = subset_end.iloc[0]
        else : 
            end = max_t

        subset_off_task_start = off_task_start[(off_task_start >= start) & (off_task_start<=end)]
        subset_seq = end_seq[(end_seq > start) & (end_seq <= end)]

        if subset_off_task_start.empty & subset_seq.empty:
            intervals.append([start, end])
        elif (subset_off_task_start.empty) & (not subset_seq.empty):
            continue
        elif (not subset_off_task_start.empty) & (not subset_seq.empty):
            off_task_stop = subset_off_task_start.iloc[0]
            seq_stop = subset_seq.iloc[0]
            if seq_stop < off_task_stop: continue
            else : 
                intervals.append([start, off_task_stop])
        else:
            stop = subset_off_task_start.iloc[0]
            start2 = off_task_end[(off_task_end >= start) & (off_task_end<=end)].iloc[-1]
            intervals.append([start, stop])
            intervals.append([start2, end])
    
    for start in off_task_end:
        subset_end = start_drinking[start_drinking>start]

        if not subset_end.empty:
            end = subset_end.iloc[0]
        
            subset_off_task_start = off_task_start[(off_task_start >= start) & (off_task_start<=end)]
            subset_seq = end_seq[(end_seq > start) & (end_seq <= end)]

            if subset_off_task_start.empty & subset_seq.empty:
                intervals.append([start, end])
            else : 
                continue
    
    moving_to_trough_zone_intervals = pd.DataFrame(intervals, columns=["start", "end"]).sort_values(by="start")
    return moving_to_trough_zone_intervals


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

def get_drinking_intervals(data):
    limit = 1000 #ms
    drinking_data = data[(data.iloc[:,1] == 5)].reset_index(drop=True)
    lick_starts = drinking_data[(drinking_data.iloc[:, 4] == 1)]

    licking_df = pd.DataFrame(columns=["Start", "End", "Type"])

    for i, row in lick_starts.iterrows():
        drinking_type = row.iloc[3]
        start = row.iloc[0]
        if i == len(drinking_data) - 1:
            end = start
        else:
            end = drinking_data.iloc[i+1,0]

        row = pd.Series([start, end, drinking_type], index=["Start", "End", "Type"])
        licking_df = pd.concat([licking_df, pd.DataFrame([row])], ignore_index=True)


    start_full_drinking_buffer = []
    end_full_drinking_buffer = []
    intervals = []
    full_licking_df = licking_df[licking_df["Type"] == 1]

    end = -1
    for i, lick in full_licking_df.iterrows():
        start = lick["Start"]

        if start < end+1:
            continue
        next_licks = licking_df[licking_df["Start"] > start]
        previous_end = lick["End"]

        for j, row in next_licks.iterrows():
            next_t = row["Start"]

            if (next_t - previous_end) > limit:
                end = previous_end
                start_full_drinking_buffer.append(start)
                end_full_drinking_buffer.append(end)
                intervals.append([start, end])

                break
            else :
                previous_end = row["End"]

    start_full_drinking_df = pd.DataFrame(start_full_drinking_buffer, columns=["time"])
    end_full_drinking_df = pd.DataFrame(end_full_drinking_buffer, columns=["time"])
    full_drinking_intervals = pd.DataFrame(intervals, columns=["start", "end"])

    mask = pd.Series(False, index=licking_df.index)
    for i, row in licking_df.iterrows():
        start = row["Start"]
        for j in range(len(start_full_drinking_df)):
            full_start = start_full_drinking_df.iloc[j, 0]
            full_end = end_full_drinking_df.iloc[j, 0]
            # If there's overlap
            if not (start > full_end or row["End"] < full_start):
                mask.iloc[i] = True
                break
    
    full_licking_df = licking_df[mask].reset_index(drop=True)

    empty_licking_df = licking_df[~mask].reset_index(drop=True)

    intervals = []
    current_start = empty_licking_df.iloc[0]['Start']
    current_end = empty_licking_df.iloc[0]['End']

    for i in range(1, len(empty_licking_df)):
        next_start = empty_licking_df.iloc[i]['Start']
        next_end = empty_licking_df.iloc[i]['End']

        if next_start > current_end + limit:
            intervals.append([current_start, current_end])
            current_start = next_start
            current_end = next_end
        else:
            current_end = max(current_end, next_end)

    intervals.append([current_start, current_end])

    empty_drinking_intervals = pd.DataFrame(intervals, columns=['start', 'end'])

    return full_drinking_intervals, empty_drinking_intervals, full_licking_df, empty_licking_df

def get_org_drinking_intervals(data):
    limit = 1000 #ms
    drinking_data = data[(data.iloc[:,1] == 5)].reset_index(drop=True)
    lick_starts = drinking_data[(drinking_data.iloc[:, 4] == 1)]

    licking_df = pd.DataFrame(columns=["Start", "End", "Type"])

    for i, row in lick_starts.iterrows():
        drinking_type = row.iloc[3]
        start = row.iloc[0]
        if i == len(drinking_data) - 1:
            end = start
        else:
            end = drinking_data.iloc[i+1,0]

        row = pd.Series([start, end, drinking_type], index=["Start", "End", "Type"])
        licking_df = pd.concat([licking_df, pd.DataFrame([row])], ignore_index=True)
    

    full_licking = licking_df[licking_df["Type"] == 1]
    intervals = []
    current_start_idx = 0
    while current_start_idx + 9 < len(full_licking):
        start = full_licking.iloc[current_start_idx]["Start"]
        end = full_licking.iloc[current_start_idx + 9]["End"]

        intervals.append([start, end])
        current_start_idx += 10

    full_drinking_intervals = pd.DataFrame(intervals, columns=["start", "end"])

    empty_licking_df = licking_df[licking_df["Type"] == 0]

    intervals = []
    current_start = empty_licking_df.iloc[0]['Start']
    current_end = empty_licking_df.iloc[0]['End']

    for i in range(1, len(empty_licking_df)):
        next_start = empty_licking_df.iloc[i]['Start']
        next_end = empty_licking_df.iloc[i]['End']

        if next_start > current_end + limit:
            intervals.append([current_start, current_end])
            current_start = next_start
            current_end = next_end
        else:
            current_end = max(current_end, next_end)

    intervals.append([current_start, current_end])

    empty_drinking_intervals = pd.DataFrame(intervals, columns=['start', 'end'])

    return full_drinking_intervals, empty_drinking_intervals, full_licking, empty_licking_df

def lever_task(moving_to_lever, sequences):
    lever_task_intervals = pd.concat([moving_to_lever, sequences]).sort_values(by="start").reset_index(drop=True)

    current_start = lever_task_intervals.iloc[0]['start']
    current_end = lever_task_intervals.iloc[0]['end']
    intervals = []
    for i in range(1, len(lever_task_intervals)):
        row = lever_task_intervals.iloc[i]
        start = row['start']
        end = row['end']

        if start <= current_end:
            current_end = max(current_end, end)
        else:
            intervals.append([current_start, current_end])
            current_start, current_end = start, end

    intervals.append([current_start, current_end])
    result = pd.DataFrame(intervals, columns=["start", "end"])

    return result

def trough_task(moving_to_trough, drinking_intervals, off_task, lever_task):
    intervals = pd.concat([moving_to_trough, drinking_intervals]).sort_values(by= "start").reset_index(drop=True)
    result = []
    current_start = intervals.iloc[0]['start']
    current_end = intervals.iloc[0]['end']
    
    for i in range(1, len(intervals)):
        next_start = intervals.iloc[i]['start']
        next_end = intervals.iloc[i]['end']

        if next_start > current_end:
            gap_contains_other_task = False

            overlapping_off = off_task[(off_task['start'] < next_start) & 
                                          (off_task['end'] > current_end)]
            if not overlapping_off.empty:
                gap_contains_other_task = True

            overlapping_lever = lever_task[(lever_task['start'] < next_start) & 
                                              (lever_task['end'] > current_end)]
            if not overlapping_lever.empty:
                gap_contains_other_task = True

            if not gap_contains_other_task:
                current_end = next_end 
            else:

                result.append([current_start, current_end])
                current_start = next_start
                current_end = next_end
        else:
            current_end = max(current_end, next_end)

    result.append([current_start, current_end])
    
    return pd.DataFrame(result, columns=["start", "end"])

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

        full_drinking_intervals, empty_drinking_intervals, full_drinking_times, empty_drinking_times = get_drinking_intervals(data)
        start_drinking_times, end_drinking_times = full_drinking_intervals.iloc[:,0], full_drinking_intervals.iloc[:, 1]
        start_empty_drinking_times, end_empty_drinking_times = empty_drinking_intervals.iloc[:,0], empty_drinking_intervals.iloc[:,1]

        seq_starts_df, seq_ends_df = action_sequences(data, start_pressing_df, end_pressing_df, enter_zone1_times, enter_zone2_times)
        seq_start_times, seq_end_times = seq_starts_df.iloc[:, 0], seq_ends_df.iloc[:,0]
        sequences = {"start": seq_start_times, "end": seq_end_times}

        licking_times = pd.concat([full_drinking_times["Start"], full_drinking_times["End"], empty_drinking_times["Start"], empty_drinking_times["End"]]).sort_values()

        off_task_intervals = off_task(enter_zone1_times, enter_zone2_times, start_pressing_times, licking_times, max_time)
        start_off_task, end_off_task = off_task_intervals.iloc[:, 0], off_task_intervals.iloc[:, 1]

        moving_to_lever_intervals = moving_to_lever_zone(seq_start_times, pd.concat([end_drinking_times, end_empty_drinking_times]).sort_values(), start_off_task, end_off_task, max_time)
        start_moving_to_lever, end_moving_to_lever = moving_to_lever_intervals.iloc[:, 0], moving_to_lever_intervals.iloc[:, 1]
        
        moving_to_trough_intervals = moving_to_trough_zone(seq_end_times, pd.concat([start_drinking_times, start_empty_drinking_times]).sort_values(), start_off_task, end_off_task, max_time)
        start_moving_to_trough, end_moving_to_trough = moving_to_trough_intervals.iloc[:, 0], moving_to_trough_intervals.iloc[:, 1]
        
        lever_task_intervals = lever_task(moving_to_lever_intervals, pd.DataFrame(sequences, columns=["start", "end"]))
        lever_task_start, lever_task_end = lever_task_intervals.iloc[:,0], lever_task_intervals.iloc[:,1]

        trough_task_intervals = trough_task(moving_to_trough_intervals, pd.concat([full_drinking_intervals, empty_drinking_intervals]), off_task_intervals, lever_task_intervals)
        trough_task_start, trough_task_end = trough_task_intervals.iloc[:,0], trough_task_intervals.iloc[:,1]



        data_dict = {
            'Pressing Start': pd.Series(start_pressing_times.values),
            'Pressing End': pd.Series(end_pressing_times.values),
            'Enter Zone 1': pd.Series(enter_zone1_times.values),
            'Enter Zone 2': pd.Series(enter_zone2_times.values),
            'Licking Full Start': pd.Series(full_drinking_times["Start"].values),
            'Licking Full End': pd.Series(full_drinking_times["End"].values),
            'Licking Empty Start': pd.Series(empty_drinking_times["Start"].values),
            'Licking Empty End': pd.Series(empty_drinking_times["End"].values),
            'Drinking Full Start': pd.Series(start_drinking_times.values),
            'Drinking Full End': pd.Series(end_drinking_times.values),
            'Drinking Empty Start': pd.Series(start_empty_drinking_times.values),
            'Drinking Empty End': pd.Series(end_empty_drinking_times.values),
            'In Lever Task Start': pd.Series(lever_task_start.values),
            'In Lever Task End': pd.Series(lever_task_end.values),
            'In Trough Task Start': pd.Series(trough_task_start.values),
            'In Trough Task End': pd.Series(trough_task_end.values),
            'Off Task Start': pd.Series(start_off_task.values),
            'Off Task End': pd.Series(end_off_task.values),
            'Sequence Start': pd.Series(seq_start_times.values),
            'Sequence End': pd.Series(seq_end_times.values),
            'Moving To Lever Start' : pd.Series(start_moving_to_lever.values),
            'Moving To Lever End' :  pd.Series(end_moving_to_lever.values),
            'Moving To Trough Start' : pd.Series(start_moving_to_trough.values),
            'Moving To Trough End' :  pd.Series(end_moving_to_trough.values),
        }
        exp_folder = os.path.dirname(file_path)
        behaviors_df = pd.DataFrame(data_dict, dtype=object)
        behaviors_df.to_csv(folder + '/' + os.path.basename(os.path.dirname(exp_folder))+ '_' + os.path.basename(exp_folder) + '_behavior_description.csv', index=False)

    return behaviors_df

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
    fig = plt.figure(figsize=(12, 4))

    start_pressing_times = data['Pressing Start'].dropna()
    end_pressing_times = data['Pressing End'].dropna()
    full_licking_times_start= data['Licking Full Start'].dropna()
    full_licking_times_end= data['Licking Full End'].dropna()
    empty_licking_times_start = data['Licking Empty Start'].dropna()
    empty_licking_times_end = data['Licking Empty End'].dropna()
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
    start_moving_to_trough = data["Moving To Trough Start"].dropna()
    end_moving_to_trough = data["Moving To Trough End"].dropna()
    enter_zone1_times = data["Enter Zone 1"].dropna()
    enter_zone2_times = data["Enter Zone 2"].dropna()

    plt.barh(y=1.25, width=np.array(end_pressing_times) - np.array(start_pressing_times), left=start_pressing_times, height=0.5, color="black", edgecolor='black', label='Press')
    
    plt.barh(y=.125, width=np.array(end_in_lever_task) - np.array(start_in_lever_task), left=start_in_lever_task, height=0.25, color="red", edgecolor='black', label='In lever task')
    plt.barh(y=.125, width=np.array(end_in_trough_task) - np.array(start_in_trough_task), left=start_in_trough_task, height=0.25, color="blue", edgecolor='black', label='In trough task')
    plt.barh(y=.125, width=np.array(end_off_task) - np.array(start_off_task), left=start_off_task, height=0.25, color="gray", edgecolor='black', label='Off task')
    
    plt.barh(y=1.75, width=np.array(end_drinking_times) - np.array(start_drinking_times), left=np.array(start_drinking_times), height=0.5, color="purple", edgecolor='black', label='Drinking full')
    plt.barh(y=1.75, width=np.array(end_empty_drinking_df) - np.array(start_empty_drinking_df), left=np.array(start_empty_drinking_df), height=0.5, color="orange", edgecolor='black', label='Drinking empty')
    
    ax = plt.gca()

    for i in range(len(full_licking_times_start)):
        start = full_licking_times_start.iloc[i]
        end = full_licking_times_end.iloc[i]
        width = end - start
        # Create an ellipse patch
        ellipse = patches.Ellipse(
            xy=(start + width/2, 2.02),  # Center of ellipse (x, y)
            width=width,  # Width of ellipse
            height=0.02,  # Height of ellipse
            angle=0,  # Angle of ellipse
            facecolor='purple',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(ellipse)

    # Draw empty licks as ovals
    for i in range(len(empty_licking_times_start)):
        start = empty_licking_times_start.iloc[i]
        end = empty_licking_times_end.iloc[i]
        width = end - start
        # Create an ellipse patch
        ellipse = patches.Ellipse(
            xy=(start + width/2, 2.02),  # Center of ellipse
            width=width,
            height=0.02,
            angle=0,
            facecolor='orange',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(ellipse)

    plt.barh(y=0.75, width=np.array(seq_end_times) - np.array(seq_start_times), left=seq_start_times, height=0.5, color="green", edgecolor='black', label='Action sequence')
    plt.barh(y=.375, width = end_moving_to_lever - start_moving_to_lever, left= start_moving_to_lever, height=0.25, color="darkred", edgecolor='black', label="moving to lever zone")
    plt.barh(y=.375, width = end_moving_to_trough - start_moving_to_trough, left= start_moving_to_trough, height=0.25, color="navy", edgecolor='black', label="moving to trough zone")
    
    ymin = -0.1
    ymax = 2.3
    # plt.vlines(np.array(seq_start_times), ymin, ymax, colors='green', linestyles="dashed", label="Press event")
    # plt.vlines(np.array(seq_end_times), ymin, ymax, colors='green', linestyles="dashed")
    # plt.vlines(np.array(enter_zone1_times), ymin, ymax, colors='blue', linestyles="dashed", label="Zone 1 event")
    # plt.vlines(np.array(enter_zone2_times), ymin, ymax, colors='red', linestyles="dashed", label="Zone 2 event")
    # plt.vlines(np.concatenate((start_drinking_times, end_drinking_times, start_empty_drinking_df, end_empty_drinking_df)), ymin, ymax, colors="black", linestyles="dashed", label="Drinking event")

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
               markeredgecolor='black', markersize=8, label='Full licks'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
               markeredgecolor='black', markersize=8, label='Empty licks')
    ]

    handles, labels = ax.get_legend_handles_labels()

    handles.extend(legend_elements)

    y_ticks = [0.125, 0.375, 0.75, 1.25, 1.75]
    y_labels = ["Task", "Moving", "Sequence", "Press", "Drinking"]
    plt.yticks(y_ticks, y_labels)
    plt.xlabel("Time (ms)")
    # plt.xlim(749000, 765000)
    plt.legend(handles=handles, loc='upper right')
    plt.tight_layout()

def plot_behavior_levels(file_list, titles = None):
    n = len(file_list)
    if n == 0:
        print("please give a file path or a list of file paths")
    elif n == 1 : 
        plot_single_experiment(file_list[0])
    else :
        fig, axes = plt.subplots(n, figsize=(14, 2*n))
        for ax, file_path, title in zip(axes, file_list, titles):
            data = load_data_from_behavior_csv(file_path)
            print("Plotting file : ", file_path)
            
            start_pressing_times = data['Pressing Start'].dropna()
            end_pressing_times = data['Pressing End'].dropna()
            full_licking_times_start= data['Licking Full Start'].dropna()
            full_licking_times_end= data['Licking Full End'].dropna()
            empty_licking_times_start = data['Licking Empty Start'].dropna()
            empty_licking_times_end = data['Licking Empty End'].dropna()
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
            start_moving_to_trough = data["Moving To Trough Start"].dropna()
            end_moving_to_trough = data["Moving To Trough End"].dropna()
            enter_zone1_times = data["Enter Zone 1"].dropna()
            enter_zone2_times = data["Enter Zone 2"].dropna()

            ax.barh(y=1.25, width=np.array(end_pressing_times) - np.array(start_pressing_times), left=start_pressing_times, height=0.5, color="black", edgecolor='black', label='Press')
            ax.barh(y=-0.25, width=np.array(end_in_lever_task) - np.array(start_in_lever_task), left=start_in_lever_task, height=0.5, color="red", edgecolor='black', label='In lever task')
            ax.barh(y=-0.25, width=np.array(end_in_trough_task) - np.array(start_in_trough_task), left=start_in_trough_task, height=0.5, color="blue", edgecolor='black', label='In trough task')
            ax.barh(y=-0.25, width=np.array(end_off_task) - np.array(start_off_task), left=start_off_task, height=0.5, color="gray", edgecolor='black', label='Off task')
            ax.barh(y=1.75, width=np.array(end_drinking_times) - np.array(start_drinking_times), left=np.array(start_drinking_times), height=0.5, color="purple", edgecolor='black', label='Drinking full')
            ax.barh(y=1.75, width=np.array(end_empty_drinking_df) - np.array(start_empty_drinking_df), left=np.array(start_empty_drinking_df), height=0.5, color="orange", edgecolor='black', label='Drinking empty')

            for i in range(len(full_licking_times_start)):
                start = full_licking_times_start[i]
                end = full_licking_times_end[i]
                width = end - start
                # Create an ellipse patch
                ellipse = patches.Ellipse(
                    xy=(start + width/2, 2.02),  # Center of ellipse (x, y)
                    width=width,  # Width of ellipse
                    height=0.02,  # Height of ellipse
                    angle=0,  # Angle of ellipse
                    facecolor='purple',
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.8
                )
                ax.add_patch(ellipse)

            # Draw empty licks as ovals
            for i in range(len(empty_licking_times_start)):
                start = empty_licking_times_start[i]
                end = empty_licking_times_end[i]
                width = end - start
                # Create an ellipse patch
                ellipse = patches.Ellipse(
                    xy=(start + width/2, 2.02),  # Center of ellipse
                    width=width,
                    height=0.02,
                    angle=0,
                    facecolor='orange',
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.8
                )
                ax.add_patch(ellipse)

            ax.barh(y=0.75, width=np.array(seq_end_times) - np.array(seq_start_times), left=seq_start_times, height=0.5, color="green", edgecolor='black', label='Action sequence')
            ax.barh(y=.25, width = end_moving_to_lever - start_moving_to_lever, left= start_moving_to_lever, height=0.5, color="darkred", edgecolor='black', label="moving to lever zone")
            ax.barh(y=.25, width = end_moving_to_trough - start_moving_to_trough, left= start_moving_to_trough, height=0.5, color="navy", edgecolor='black', label="moving to trough zone")

            # if ax == axes[-1]:
            #     ax.set_xlabel("Time (ms)")
            y_ticks = [-0.25, 0.25, 0.75, 1.25, 1.75]
            y_labels = ["Task", "Moving", "Sequence", "Press", "Drinking"]
            ax.set_yticks(y_ticks, y_labels)
            ax.set_title(title, loc='left', fontweight= 'bold')

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show()
    return


def load_data_from_behavior_csv(file_path):
    # Loads the data from behavior description csv. This csv is saved using the above save_behavior_description(args) function
    data = pd.read_csv(file_path)
    return data

def fuse_drinking():
    sessions = ["Exp 014", "Exp 015", "Exp 016", "Exp 017"]
    mice = ["M2", "M4", "M15"]
    for mouse in mice:

        folder_path = f"behavioral_data\\behavior descriptions\\final_description\\{mouse}"
        save_folder = f"behavioral_data\\behavior descriptions\\fused_drinking\\{mouse}"
        file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        sessions_file_list = []

        for file_path in file_list:
            if any(exp in file_path for exp in sessions):
                sessions_file_list.append(file_path)

        for file in sessions_file_list:
            print("Fusing file: ", file)
            if not os.path.exists(file):
                continue
            file_name = os.path.basename(file)
            df = load_data_from_behavior_csv(file)
            new_df = df.copy()

            drinking_start = pd.concat([
                df['Drinking Full Start'].dropna(),
                df['Drinking Empty Start'].dropna()
            ]).sort_values()

            drinking_end = pd.concat([
                df['Drinking Full End'].dropna(),
                df['Drinking Empty End'].dropna()
            ]).sort_values()

            new_df['Drinking Start'] = drinking_start.reset_index(drop=True)
            new_df['Drinking End'] = drinking_end.reset_index(drop=True)

            columns_to_drop = [
                'Drinking Full Start', 'Drinking Full End',
                'Drinking Empty Start', 'Drinking Empty End'
            ]
            new_df = new_df.drop(columns=columns_to_drop)
            new_df.to_csv(os.path.join(save_folder, file_name), index=False)

def time_duration_session():
    mice = ["M2", "M4", "M15"]
    exp_list = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "6th FR1"]
    total_session_distributions = [np.zeros(6) for mouse in mice]
    for i, mouse in enumerate(mice):
        path_file = f"behavioral_data\\paths\\paths_dat\\{mouse}_dat_exp10_to_17.csv"
        paths = pd.read_csv(path_file)["File"].values
        file_list = np.concat([paths[:5], [paths[6]]])
        for j, file in enumerate(file_list):
            print(file)
            data = load_data_from_dat(file)
            drinking_data = data[data.iloc[:,1] == 5]
            total_distributions = np.max(drinking_data.iloc[:,7])
            total_session_distributions[i][j] = total_distributions

    x = np.arange(len(exp_list))
    plt.figure(figsize=(10, 6))

    # Use better colors and add markers
    for i, mouse_data in enumerate(total_session_distributions):
        plt.plot(x, mouse_data, label=mice[i], marker='o', linewidth=2)

    plt.xticks(x, exp_list, rotation=45)  # Rotate labels if needed
    plt.xlabel("Session number", fontsize=12)
    plt.ylabel("Number of liquid distributions", fontsize=12)
    plt.grid(alpha=0.3, linestyle='--')  # Subtle grid
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.show()


def behavior_episodes():
    mice = ["M2", "M4", "M15"]
    exp_list = ["1st FR1", "2nd FR1", "3rd FR1", "4th FR1", "5th FR1", "6th FR1"]
    behavior_list = ['Sequence', 'Moving To Trough', 'Locomotion To Trough', 'Drinking Full', 'Moving To Lever', 'Locomotion To Lever', 'Drinking Empty', 'Off Task']
    
    all_durations = {behavior: [] for behavior in behavior_list}

    for mouse in mice:
        folder = f"behavioral_data\\behavior descriptions\\final_description\\{mouse}"
        file_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

        for f in file_list:
            data = load_data_from_behavior_csv(f)

            for behavior in behavior_list:
                try:
                    episodes = data[[behavior + " Start", behavior + " End"]].dropna()
                    durations = episodes.iloc[:, 1] - episodes.iloc[:, 0]
                    if not durations.empty:
                        all_durations[behavior].extend(durations.tolist())
                except KeyError:
                    continue

    mean_data = {
    "Behavior": [],
    "Mean Duration (s)": []
    }

    for behavior in behavior_list:
        durations = all_durations[behavior]
        if durations:
            mean_duration = np.mean(durations)/1000
            mean_data["Behavior"].append(behavior)
            mean_data["Mean Duration (s)"].append(mean_duration)

    df_mean = pd.DataFrame(mean_data)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_mean, x="Behavior", y="Mean Duration (s)", palette=["green", "navy", "cornflowerblue", "purple", "darkred", "orangered", "orange", 'gray'])
    plt.xticks(rotation=45, ha='right')
    plt.yticks(np.arange(26, step=1))
    plt.ylabel("Mean duration (s)")
    plt.tight_layout()
    plt.show()
            

def main():
    # behavior_episodes()
    # time_duration_session()
    # fuse_drinking()
    # fuse_movements()
    # compute_task_intervals()

    #COMPUTE BEHAVIOR LEVELS DURING EXPERIMENTS
    # for mouse in ["M2", "M4", "M15"]:
    #     file_list = pd.read_csv(f".\\behavioral_data\\paths\\paths_dat\\{mouse}_dat_exp10_to_17.csv")["File"]
    #     compute_behavior_description(file_list, f".\\behavioral_data\\behavior descriptions\\complete_description\\{mouse}")

    #COMPUTE METRICS ON SAVED BEHAVIOR DESCRIPTIONS
    # file_list = [r'.\behavioral_data\behavior descriptions\full session\M2\\' + f for f in os.listdir(r'.\behavioral_data\behavior descriptions\full session\M2\\')]
    # compute_partial_behavior_description([r"C:\Users\maxge\OneDrive - Université Libre de Bruxelles\MA2\Mémoire\master_thesis\behavioral_data\behavior descriptions\full session\M2 - Jun24_Exp 010_behavior_description.csv"], r'C:\Users\maxge\OneDrive - Université Libre de Bruxelles\MA2\Mémoire\master_thesis\behavioral_data\behavior descriptions\partial sessions', 4)
    # compute_behavior_description([r"P:\Ca2+ Data\M2 - Jun24\Exp 013\M2_240620_FR1_4good_01.dat"], r"behavioral_data\behavior descriptions")

    #PLOT ALL SESSION 1 BY 1
    # for mouse in ["M2", "M4", "M15"]:
    #     folder = f".\\behavioral_data\\behavior descriptions\\complete_description\\{mouse}"
    #     file_list = [os.path.join(folder, f) for f in os.listdir(folder)]
    #     for file in file_list:
    #         plot_single_experiment(file)
    #         plt.show()
    
    #PLOT SINGLE SESSION
    plot_single_experiment(r"behavioral_data\behavior descriptions\final_description\M2\M2 - Jun24_Exp 010_behavior_description.csv")
    plot_single_experiment("segment_1.csv")
    plot_single_experiment("segment_2.csv")
    plot_single_experiment("segment_3.csv")
    plot_single_experiment("segment_4.csv")
    plt.show()

    #PLOT MULTIPLE SESSIONS
    # FR1 = ["Exp 016", "Exp 017"] #, "Exp 014", "Exp 016"]
    # mice = ["M2", "M4", "M15"]
    # titles = ['Sixth FR1 session', 'Complete devaluation']
    # folder_path = r"behavioral_data\behavior descriptions\final_description\M4"
    # file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    # FR1_file_list = []

    # for file_path in file_list:
    #     if any(exp in file_path for exp in FR1):
    #         FR1_file_list.append(file_path)
    # plot_behavior_levels(FR1_file_list, titles)


    # plot_behavior_levels([r"C:\Users\maxge\OneDrive - Université Libre de Bruxelles\MA2\Mémoire\master_thesis\behavioral_data\behavior descriptions\full session\M2 - Jun24_Exp 010_behavior_description.csv"])
    return

if __name__ == "__main__": 
    main()