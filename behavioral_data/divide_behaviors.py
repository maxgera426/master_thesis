import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import os

def divide_behavior_descriptions(file_path: str, n: int, max_t: float) -> List[pd.DataFrame]:

    df = pd.read_csv(file_path)

    segment_duration = max_t / n
    segment_boundaries = [(i * segment_duration, (i + 1) * segment_duration) 
                         for i in range(n)]

    segmented_dfs = []
    
    for seg_idx, (seg_start, seg_end) in enumerate(segment_boundaries):
        segment_df = pd.DataFrame(columns=df.columns)

        for idx, row in df.iterrows():
            new_row = {}

            new_row[df.columns[0]] = len(segment_df)

            for col_idx in range(1, len(df.columns), 2):
                if col_idx + 1 >= len(df.columns):
                    break
                    
                start_col = df.columns[col_idx]
                end_col = df.columns[col_idx + 1]
                
                start_time = row[start_col]
                end_time = row[end_col]

                if pd.isna(start_time) and pd.isna(end_time):
                    new_row[start_col] = np.nan
                    new_row[end_col] = np.nan
                    continue

                if (pd.isna(start_time) or start_time >= seg_end or 
                    pd.isna(end_time) or end_time <= seg_start):
                    new_row[start_col] = np.nan
                    new_row[end_col] = np.nan
                    continue

                adjusted_start = max(start_time, seg_start) if not pd.isna(start_time) else seg_start
                adjusted_end = min(end_time, seg_end) if not pd.isna(end_time) else seg_end

                relative_start = adjusted_start - seg_start
                relative_end = adjusted_end - seg_start
                
                new_row[start_col] = relative_start
                new_row[end_col] = relative_end

            has_valid_behavior = any(
                not pd.isna(new_row.get(df.columns[i], np.nan)) 
                for i in range(1, len(df.columns))
            )
            
            if has_valid_behavior:
                segment_df = pd.concat([segment_df, pd.DataFrame([new_row])], ignore_index=True)

        if not segment_df.empty:
            segment_df.reset_index(drop=True, inplace=True)
            segment_df.iloc[:, 0] = range(len(segment_df))
        
        segmented_dfs.append(segment_df)
    
    return segmented_dfs

def save_segmented_data(segmented_dfs: List[pd.DataFrame], 
                       original_file_path: str, 
                       output_dir: str = None) -> List[str]:
    
    if output_dir is None:
        output_dir = os.path.dirname(original_file_path)

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(original_file_path))[0]
    
    saved_files = []
    for i, df in enumerate(segmented_dfs):
        output_file = os.path.join(output_dir, f"{base_name}_segment_{i+1}.csv")
        df.to_csv(output_file, index=False)
        saved_files.append(output_file)
    
    return saved_files

if __name__ == "__main__":
    duration = [1240046,1213640,1240046]
    mice = ["M2", "M4", "M15"]
    for i, mouse in enumerate(mice):

        file_path = f"behavioral_data\\behavior descriptions\\final_description\\{mouse}\\{mouse} - Jun24_Exp 017_behavior_description.csv"
        save_folder = f"behavioral_data\\behavior descriptions\\divided_descriptions\\{mouse}"
        segments = divide_behavior_descriptions(file_path, n=3, max_t=duration[i])

        saved_files = save_segmented_data(segments, file_path, save_folder)
        print(f"Saved {len(saved_files)} segment files")