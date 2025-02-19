import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from time import time


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

    # Compute angle of the longest edge with vertical
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    rotation_angle = np.arctan2(dx, dy)

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

    width, height = rect_coords[2] - rect_coords[0]
    f_height = height/22
    mouse_length = 5*f_height
    lever_zone = np.array([[x_min, y_max - mouse_length], [x_min + width / 2, y_max - mouse_length], [x_min + width / 2, y_max], [x_min, y_max]])
    trough_zone = np.array([[x_min + width / 2, y_max - mouse_length], [x_max, y_max - mouse_length], [x_max, y_max], [x_min + width / 2, y_max]])

    top_right_corner_zone = np.array([[x_max - 1/5*width, y_min], [x_max, y_min], [x_max, y_min + 1/5*width], [x_max - 1/5*width, y_min + 1/5*width]])
    top_left_corner_zone = np.array([[x_min, y_min], [x_min + 1/5*width, y_min], [x_min + 1/5*width, y_min + 1/5*width], [x_min, y_min + 1/5*width]])

    fig.add_trace(go.Scatter(
        x=np.append(lever_zone[:, 0], lever_zone[:, 0][0]),
        y=np.append(lever_zone[:, 1], lever_zone[:, 1][0]),
        mode='lines',
        name='Lever Zone',
        line=dict(color='red')
    ))

    fig.add_trace(go.Scatter(
        x=np.append(trough_zone[:, 0],trough_zone[:, 0][0]),
        y=np.append(trough_zone[:, 1],trough_zone[:, 1][0]),
        mode='lines',
        name='Trough Zone',
        line=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=np.append(top_right_corner_zone[:, 0], top_right_corner_zone[:, 0][0]),
        y=np.append(top_right_corner_zone[:, 1], top_right_corner_zone[:, 1][0]),
        mode='lines',
        name='Top Right Corner Zone',
        line=dict(color='purple')
    ))

    fig.add_trace(go.Scatter(
        x=np.append(top_left_corner_zone[:, 0], top_left_corner_zone[:, 0][0]),
        y=np.append(top_left_corner_zone[:, 1], top_left_corner_zone[:, 1][0]),
        mode='lines',
        name='Top Left Corner Zone',
        line=dict(color='orange')
    ))
    # Show the figure
    fig.show()


# def positions_time(data, body_parts, likelihood_threshold=0.95):
#     # Function to create data frame containing all significant positions in the order specified by body_parts list
    
#     for part in body_parts:


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
    

def main():
    # Load the data
    file_path = r"P:\Ca2+ Data\F5\Exp012\F5_230124_day2DLC_resnet50_Ethogram operantApr12shuffle1_100000.csv"
    column_names = ['frame number', 'neck_x', 'neck_y', 'neck_likelihood', 'body1_x', 'body1_y', 'body1_likelihood',
                    'body2_x', 'body2_y', 'body2_likelihood', 'body3_x', 'body3_y', 'body3_likelihood', 'tail_start_x',
                    'tail_start_y', 'tail_start_likelihood', 'tail_end_x', 'tail_end_y', 'tail_end_likelihood',
                    'earR_x', 'earR_y', 'earR_likelihood', 'earL_x', 'earL_y', 'earL_likelihood', 'camera_x',
                    'camera_y', 'camera_likelihood', 'nose_x', 'nose_y', 'nose_likelihood', 'pawR_x', 'pawR_y',
                    'pawR_likelihood', 'pawL_x', 'pawL_y', 'pawL_likelihood']
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, skiprows=3, header=None, names=column_names)
    else:
        print("File not found")
        return

    likelihood_threshold = 0.95
    body_part = 'camera'
    data['time'] = data['frame number'] / 20 * 1000 # ms

    body_parts = ['neck', 'body1', 'body2', 'body3', 'tail_start', 'tail_end', 'camera', 'earR', 'earL', 'nose', 'pawR', 'pawL']
    total_data = pd.DataFrame(columns=['all_x', 'all_y'])
    for part in body_parts:
        part_data = data[(data[f'{part}_likelihood'] > likelihood_threshold)][[f'{part}_x', f'{part}_y']]
        part_data.columns = ['all_x', 'all_y']
        total_data = pd.concat([total_data, part_data], ignore_index=True)
    # Find the corners of the datafrance fffr
    rect_coords = find_rectangle(total_data).T
    filtered_data = data[(data[f'{body_part}_likelihood'] > likelihood_threshold)]
    # Rotate the data
    rect_coords, rotated_data = rotate_data(filtered_data, rect_coords)

    plot_positions(rotated_data, body_part, rect_coords)
    # positions_time(data, mean_pos=mean_positions, body_part=body_part, likelihood_threshold=likelihood_threshold)
    zone_intervals = get_zone(rotated_data, body_part, rect_coords)
    return zone_intervals


if __name__ == "__main__":
    main()
