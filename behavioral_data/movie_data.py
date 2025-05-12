import cv2
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import seaborn as sns


# This script handles the data extracted from the movie of the mouse movements taken during the experiments
def compute_mean_position(data, body_part="all", likelihood_threshold=0.95):
    if body_part == 'all':
        # Compute the mean position of all the body parts
        body_parts = ['neck', 'body1', 'body2', 'body3', 'tail_start', 'tail_end', 'camera', 'earR', 'earL', 'nose', 'pawR', 'pawL']
        mean_positions = {}
        for part in body_parts:
            filtered_data = data[data[f'{part}_likelihood'] > likelihood_threshold]
            if len(filtered_data) > 0:
                mean_positions[part] = (np.mean(filtered_data[f'{part}_x']), np.mean(filtered_data[f'{part}_y']))
        mean_positions["global"] = (np.mean([pos[0] for pos in mean_positions.values()]), np.mean([pos[1] for pos in mean_positions.values()]))
        return mean_positions


def find_rectangle(filtered_data):
    # Filter points by likelihood threshold
    points = filtered_data[['all_x', 'all_y']]
    # Get Convex Hull
    hull = ConvexHull(points)
    hull_points = points.iloc[hull.vertices]

    # Convert hull points to a Shapely Polygon
    polygon = Polygon(hull_points)

    # Get the minimum area rectangle
    min_area_rect = polygon.minimum_rotated_rectangle

    # Extract the rectangle's coordinates
    rect_coords = np.array(min_area_rect.exterior.coords)  # Remove duplicate last point
    return rect_coords

def rotate_data(data, rect):
    x = rect[0]
    y = rect[1]
    z = np.ones_like(x)
    rect = np.vstack((rect, z))
    body_parts = ['neck', 'body1', 'body2', 'body3', 'tail_start', 'tail_end', 'camera', 'earR', 'earL', 'nose', 'pawR', 'pawL']
    

    A = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])

    C_x = (1 / (6 * A)) * np.sum((x[:-1] + x[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))
    C_y = (1 / (6 * A)) * np.sum((y[:-1] + y[1:]) * (x[:-1] * y[1:] - x[1:] * y[:-1]))

    # Find the longest edge
    edges = []
    for i in range(len(rect[0]) - 1):
        p1 = rect[:, i]
        p2 = rect[:, i + 1]
        dist = np.linalg.norm(p2 - p1)
        edges.append((dist, p1, p2))

    longest_edge = max(edges, key=lambda x: x[0])
    p1, p2 = longest_edge[1], longest_edge[2]

    # Compute angle of the longest edge with horizontal
    dx, dy = np.abs(p2[0] - p1[0]), np.abs(p2[1] - p1[1])
    rotation_angle = np.arctan2(dy, dx) # if you want angle with vertical swap dy and dx

    if p1[1] < p2[1]:
        rotation_angle = np.abs(rotation_angle)
    elif p1[1] > p2[1]:
        rotation_angle = -np.abs(rotation_angle)

    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])

    translation_matrix = np.array([
        [1, 0, C_x],
        [0, 1, C_y],
        [0, 0, 1]
    ])

    reverse_translation_matrix = np.array([
        [1, 0, -C_x],
        [0, 1, -C_y],
        [0, 0, 1]
    ])
    transformation_matrix = translation_matrix @ rotation_matrix @ reverse_translation_matrix
    rotated_rect = transformation_matrix @ rect
    #rotate the data
    rotated_data = data.copy()
    for body_part in body_parts:
        pos = data[[f'{body_part}_x', f'{body_part}_y']].T.values
        pos = np.vstack((pos, np.ones_like(pos[0])))
        rotated_pos = transformation_matrix @ pos
        rotated_data[[f'{body_part}_x', f'{body_part}_y']] = rotated_pos[0:2].T

    final_rect = np.array([rotated_rect[0], rotated_rect[1]])
    return final_rect, rotated_data

def plot_positions(data, body_part, rect_coords):
    
    # Create the figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data[f'{body_part}_x'],
        y=data[f'{body_part}_y'],
        mode='markers',
        name='Data',
        marker=dict(color='blue'),
        # hovertext=filtered_data.apply(lambda row: f"Frame: {row['frame number']}<br>Likelihood: {row['camera_likelihood']}", axis=1)
    ))
    
    # Add the final corners
    fig.add_trace(go.Scatter(
        x=rect_coords[0],
        y=rect_coords[1],
        mode='lines+markers',
        name='Corners',
        marker=dict(color='black', size=12, symbol='x')
    ))

    # Update layout
    fig.update_layout(
        title=f'{body_part} position',
        xaxis_title='X',
        yaxis_title='Y',
        yaxis=dict(
            autorange='reversed',
            scaleanchor="x",
            scaleratio=1,
        ),
    )

    x_min, y_min = rect_coords[0].min(), rect_coords[1].min()
    x_max, y_max = rect_coords[0].max(), rect_coords[1].max()
    rect_coords = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
    fig.show()


def get_zone(data, body_part, rect_coords):
    x_min, y_min = rect_coords[0].min(), rect_coords[1].min()
    x_max, y_max = rect_coords[0].max(), rect_coords[1].max()
    rect_coords = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

    width, height = rect_coords[2] - rect_coords[0]
    
    lever_zone = np.array([[x_min, y_min + 3/4*height], [x_min + width / 2, y_min + 3/4*height], [x_min + width / 2, y_max], [x_min, y_max]])
    trough_zone = np.array([[x_min + width / 2, y_min + 3/4*height], [x_max, y_min + 3/4*height], [x_max, y_max], [x_min + width / 2, y_max]])

    data['zone'] = np.where((data[f'{body_part}_x'] > lever_zone[0][0]) & (data[f'{body_part}_x'] < lever_zone[1][0]) & (data[f'{body_part}_y'] > lever_zone[0][1]) & (data[f'{body_part}_y'] < lever_zone[2][1]), 'lever_zone', 'other')
    data['zone'] = np.where((data[f'{body_part}_x'] > trough_zone[0][0]) & (data[f'{body_part}_x'] < trough_zone[1][0]) & (data[f'{body_part}_y'] > trough_zone[0][1]) & (data[f'{body_part}_y'] < trough_zone[2][1]), 'trough_zone', data['zone'])
    
    data['zone_change'] = data['zone'].ne(data['zone'].shift())
    data['group'] = data['zone_change'].cumsum()
    # pd.DataFrame.to_csv(data, 'data.csv', columns=['camera_x',
    #                 'camera_y', 'camera_likelihood', 'time', 'zone', 'zone_change', 'group'])

    zone_intervals = data.groupby(['group', 'zone']).agg(start_time=('time', 'first'), end_time=('time', 'last')).reset_index().drop(columns=['group'])
    return zone_intervals

def all_points(data, body_parts, likelihood = 0.95):
    total_data = pd.DataFrame(columns=['all_x', 'all_y'])
    for part in body_parts:
        part_data = data[(data[f'{part}_likelihood'] > likelihood)][[f'{part}_x', f'{part}_y']]
        part_data.columns = ['all_x', 'all_y']
        total_data = pd.concat([total_data, part_data], ignore_index=True)
    
    return total_data

def load_data(file_path):
    column_names = ['frame number', 'neck_x', 'neck_y', 'neck_likelihood', 'body1_x', 'body1_y', 'body1_likelihood',
                    'body2_x', 'body2_y', 'body2_likelihood', 'body3_x', 'body3_y', 'body3_likelihood', 'tail_start_x',
                    'tail_start_y', 'tail_start_likelihood', 'tail_end_x', 'tail_end_y', 'tail_end_likelihood',
                    'earR_x', 'earR_y', 'earR_likelihood', 'earL_x', 'earL_y', 'earL_likelihood', 'camera_x',
                    'camera_y', 'camera_likelihood', 'nose_x', 'nose_y', 'nose_likelihood', 'pawR_x', 'pawR_y',
                    'pawR_likelihood', 'pawL_x', 'pawL_y', 'pawL_likelihood']
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, skiprows=3, header=None, names=column_names)
        data['Time'] = data['frame number'] / 20 * 1000 # ms
        return data
    else:
        print("File not found")
        return

def moving(positions, units, total_t = 1200000):
    x_body1 = positions[:, 0]
    y_body1 = positions[:, 1]
    time = positions[:, 2]
    real_time = np.arange(0, total_t + 1, 50)
    delta_t = np.diff(real_time)
    x_interp = np.interp(real_time, time, x_body1)
    y_interp = np.interp(real_time, time, y_body1)

    smooth_x = savgol_filter(x_interp, 31, 3)
    smooth_y = savgol_filter(y_interp, 31, 3)

    plt.figure()
    # X coordinates - using same color (blue) with different line styles
    plt.plot(real_time/1000, x_interp * units, "--", color='blue', label="x Position")
    plt.plot(real_time/1000, smooth_x * units, "-", color='blue', label="Smooth x Position")

    # Y coordinates - using same color (orange) with different line styles
    plt.plot(real_time/1000, y_interp * units, "--", color='orange', label="y Position")  # Note: this should be y_interp not x_interp again
    plt.plot(real_time/1000, smooth_y * units, "-", color='orange', label="Smooth y Position")

    plt.xlabel('Time (seconds)')
    plt.ylabel('Position (pixels)')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


    v_t = np.sqrt(np.diff(smooth_x)**2 + np.diff(smooth_y)**2)/(delta_t[0])*1000*units # cm/s

    velocity_threshold = 1.5
    is_moving = v_t >= velocity_threshold
    transitions = np.diff(is_moving.astype(int))
    start_moving = np.where(transitions == 1)[0] + 1
    stop_moving = np.where(transitions == -1)[0] + 1
    
    if len(start_moving) > 0 and len(stop_moving) > 0:
        if start_moving[0] > stop_moving[0]:
            start_moving = np.insert(start_moving, 0, 0)
        if start_moving[-1] > stop_moving[-1]:
            stop_moving = np.append(stop_moving, len(is_moving))
    
    # Calcul des durées et distances de chaque segment de mouvement
    movement_segments = []
    if len(start_moving) > 0 and len(stop_moving) > 0:
        min_length = min(len(start_moving), len(stop_moving))
        for i in range(min_length):
            start_idx = start_moving[i]
            stop_idx = stop_moving[i]
            segment_duration = (real_time[stop_idx] - real_time[start_idx]) / 1000  # en secondes
            
            # Distance parcourue pendant ce segment
            dx = x_interp[stop_idx] - x_interp[start_idx]
            dy = y_interp[stop_idx] - y_interp[start_idx]
            distance = np.sqrt(dx**2 + dy**2) * units
            
            # Vitesse moyenne pendant ce segment
            avg_velocity = distance / segment_duration if segment_duration > 0 else 0
            if (segment_duration >= 0.5) & (distance >= 2):
                movement_segments.append((
                    real_time[start_idx]/1000,  # en secondes
                    real_time[stop_idx]/1000,    # en secondes
                    segment_duration,             # en secondes
                    distance,                     # en unités fournies
                    avg_velocity              # unités/seconde
                ))
    
    # Afficher la vitesse instantanée avec le seuil
    plt.figure(figsize=(14, 8))
    plt.subplot(211)
    plt.plot(real_time[1:]/1000, v_t, label="Instantaneous velocity")
    plt.axhline(y=velocity_threshold, color='r', linestyle='--', label=f'Threshold: {velocity_threshold:.4f} cm/s')
    plt.xlabel('Time (secondes)')
    plt.ylabel('Velocity (cm/s)')
    # plt.title('Vitesse instantanée et seuil de détection')
    plt.legend()
    plt.xlim(30, 115)
    plt.grid(True)
    
    # # Visualiser les positions avec les périodes de mouvement surlignées
    plt.subplot(212)
    plt.plot(real_time/1000, smooth_x * units, label="x position")
    plt.plot(real_time/1000, smooth_y * units, label="y position")
    
    # Surligner les périodes de mouvement
    for i, segment in enumerate(movement_segments):
            if i == 0:
                plt.axvspan(segment[0], segment[1], 
                        alpha=0.2, color='green', label="Movement intervals")
            else: 
                plt.axvspan(segment[0], segment[1], 
                        alpha=0.2, color='green')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Position (cm)')
    # plt.title('Positions avec périodes de mouvement surlignées')
    plt.legend(loc = "upper right")
    plt.grid(True)
    plt.xlim(30, 115)
    plt.tight_layout()
    plt.show()
    
    return movement_segments

def get_file_list(folder_path):
    file_list = []
    experiment_nums = [f"0{i}" for i in range(10,18)]

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("filtered.csv"):
                if any(f"{num}" in root for num in experiment_nums):
                    file_list.append(os.path.join(root, file))
    return file_list

def save_movement_description(list, exp_num):
    columns = ["Start", "Stop", "Duration", "Distance", "Avg velocity"]
    movement_df = pd.DataFrame(list, columns=columns)
    save_file = r"behavioral_data\behavior descriptions\movement_description\M15\\" + exp_num + "_movement_segments.csv"
    movement_df.to_csv(save_file, index=False)

def plot_n_points():
    mice = ["M2", "M4", "M15"]
    likelihood_threshold = 0.8
    body_parts = ['neck', 'body1', 'body2', 'body3', 'tail_start', 'tail_end', 'pawR', 'pawL']
    points = np.zeros((len(body_parts), len(mice)*8))
    for i, mouse in enumerate(mice):
        dir = f"P:\\Ca2+ Data\\{mouse} - Jun24\\"
        file_list = get_file_list(dir)
        for j, file in enumerate(file_list):
            print(file)
            data = load_data(file)
            for k, part in enumerate(body_parts):
                col = i*8 + j
                n_points = len(data[(data[f'{part}_likelihood'] > likelihood_threshold)])
                points[k, col] = n_points
    
    plt.figure()
    sns.boxplot(data=points.T, palette=["darkviolet", "slateblue", "cornflowerblue", "lightskyblue", "turquoise", "mediumspringgreen"])
    plt.xticks(range(len(body_parts)), body_parts)
    plt.ylabel("Number of points")
    plt.show()
        



def main():
    # plot_n_points()
    # Load the data
    cage_dimensions = (22, 15.5) # cm 
    dir = r"P:\Ca2+ Data\M2 - Jun24\\"
    likelihood_threshold = 0.95
    body_part = 'body1'
    body_parts = ['neck', 'body1', 'body2', 'body3', 'tail_start', 'tail_end', 'camera', 'earR', 'earL', 'nose', 'pawR', 'pawL']
    file_list = get_file_list(dir)

    for file_path in file_list:
        print("Processing: ", file_path)
        video_path = file_path.replace(".csv", "_labeled.mp4")
        data = load_data(file_path)
        total_t = np.max(data["Time"])
        total_data = all_points(data, body_parts, likelihood_threshold)
        filtered_data = data[(data['camera_likelihood'] > likelihood_threshold)]
        rect_coords = find_rectangle(total_data).T

        ## Show video frame
        # cap = cv2.VideoCapture(video_path)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        # ret, frame = cap.read()
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # fig,ax = plt.subplots(figsize = (12,8))
        # ax.imshow(frame_rgb)
        # ax.plot(rect_coords[0], rect_coords[1], color="red")
        # plt.xlabel("x (pixel)")
        # plt.ylabel("y (pixel)")
        # plt.show()

        # Find the corners of the cage
        rect_coords = find_rectangle(total_data).T
        likelihood_threshold = 0.80 # smaller to include more points
        filtered_data = data[(data[f'{body_part}_likelihood'] > likelihood_threshold)]

        # Rotate the data to compensate fisheye effect
        rect_coords, rotated_data = rotate_data(filtered_data, rect_coords)


        x_diff = max(rect_coords[0]) - min(rect_coords[0])
        y_diff = max(rect_coords[1]) - min(rect_coords[1])
        length = max(x_diff, y_diff)
        width = min(x_diff, y_diff)
        x_factor = cage_dimensions[0] / length
        y_factor = cage_dimensions[1] / width
        pixel_to_cm = np.mean([x_factor, y_factor])
        

        movement_segments = moving(rotated_data[[f"{body_part}_x", f'{body_part}_y', 'Time']].values, pixel_to_cm, total_t)

        # exp_num = os.path.basename(os.path.dirname(file_path))
        # save_movement_description(movement_segments, exp_num)
    return 0


if __name__ == "__main__":
    main()
