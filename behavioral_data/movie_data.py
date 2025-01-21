import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

# This script handles the data extracted from the movie of the mouse movements taken during the experiments 
def compute_mean_position(data, body_part = "all", likelihood_threshold=0.95):
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
    
def plot_positions(data, mean_pos, body_part='all', likelihood_threshold=0.95):
    lever_filtered_data = data[(data[f'{body_part}_likelihood'] > likelihood_threshold) & (data[f'{body_part}_x'] < mean_pos[body_part][0])]
    trough_filtered_data = data[(data[f'{body_part}_likelihood'] > likelihood_threshold) & (data[f'{body_part}_x'] > mean_pos[body_part][0])]

    # Plot the data
    # Create the figure
    fig = go.Figure()

    # Add lever data
    fig.add_trace(go.Scatter(
        x=lever_filtered_data['camera_x'],
        y=lever_filtered_data['camera_y'],
        mode='markers',
        name='Lever',
        marker=dict(color='blue'),
        hovertext=lever_filtered_data.apply(lambda row: f"Frame: {row['frame number']}<br>Likelihood: {row['camera_likelihood']}", axis=1)
    ))

    # Add trough data
    fig.add_trace(go.Scatter(
        x=trough_filtered_data['camera_x'],
        y=trough_filtered_data['camera_y'],
        mode='markers',
        name='Trough',
        marker=dict(color='green'),
        hovertext=trough_filtered_data.apply(lambda row: f"Frame: {row['frame number']}<br>Likelihood: {row['camera_likelihood']}", axis=1)
    ))

    # Add vertical line for mean position
    fig.add_vline(x=mean_pos[body_part][0], line_dash="dash", line_color="red")

    # Update layout
    fig.update_layout(
        title=f'{body_part} position',
        xaxis_title='X',
        yaxis_title='Y',
        yaxis=dict(autorange='reversed')
    )

    # Show the figure
    fig.show()

def positions_time(data, mean_pos, body_part='all', likelihood_threshold=0.95):
    # Filter the data
    filtered_data = data[data[f'{body_part}_likelihood'] > likelihood_threshold]

    # Create the figure
    fig = go.Figure()

    # Add the data
    fig.add_trace(go.Scatter(
        x=(filtered_data['frame number'])/20,
        y=filtered_data[f'{body_part}_x'],
        mode='lines',
        name='X position',
        line=dict(color='blue'),
        hovertext=filtered_data.apply(lambda row: f"Y: {row[f'{body_part}_y']}", axis=1)
    ))

    fig.add_hline(y=mean_pos[body_part][0], line_dash="dash", line_color="red")

    fig.update_layout(
        title=f'{body_part} position',
        xaxis_title='Time (s)',
        yaxis_title='X position'
    )

    fig.show()
    
def main():
    # Load the data
    file_path = r"P:\Ca2+ Data\F5\Exp 011\F5_230124_day2DLC_resnet50_Ethogram operantApr12shuffle1_100000.csv"
    column_names = ['frame number','neck_x','neck_y','neck_likelihood','body1_x','body1_y','body1_likelihood','body2_x','body2_y','body2_likelihood','body3_x','body3_y','body3_likelihood','tail_start_x','tail_start_y','tail_start_likelihood','tail_end_x','tail_end_y','tail_end_likelihood','earR_x','earR_y','earR_likelihood','earL_x','earL_y','earL_likelihood','camera_x','camera_y','camera_likelihood','nose_x','nose_y','nose_likelihood','pawR_x','pawR_y','pawR_likelihood','pawL_x','pawL_y','pawL_likelihood']
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, skiprows=3, header=None, names=column_names)
    else:
        print("File not found")
        return

    likelihood_threshold = 0.95
    mean_positions = compute_mean_position(data, body_part='all', likelihood_threshold=likelihood_threshold)
    body_part = 'camera'

    plot_positions(data, mean_positions, body_part=body_part, likelihood_threshold=likelihood_threshold)
    positions_time(data,mean_pos=mean_positions ,body_part=body_part, likelihood_threshold=likelihood_threshold)
    

if __name__ == "__main__":
    main()

    