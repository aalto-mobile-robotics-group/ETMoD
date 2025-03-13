import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

def Draw_cluster_loc(data, membership):
    # Extract positions from the data (x and y coordinates)
    x_positions = data.iloc[:, 2].to_numpy()  # Third column
    y_positions = data.iloc[:, 3].to_numpy()  # Fourth column

    # Prepare a figure with a compact size
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)  # High resolution and compact size

    # Unique cluster labels
    unique_clusters = np.unique(membership)

    # Define colors for each cluster using a colormap with many distinct colors
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(unique_clusters)))

    # Iterate through each cluster to plot points and draw circles
    for cluster, color in zip(unique_clusters, colors):
        mask = membership == cluster  # Get points belonging to the cluster
        cluster_x = x_positions[mask]
        cluster_y = y_positions[mask]
        
        # Plot the points for this cluster
        ax.scatter(cluster_x, cluster_y, c=[color], alpha=0.7, s=10)
        
        # Calculate the circle to frame all the points
        cluster_center_x = np.mean(cluster_x)
        cluster_center_y = np.mean(cluster_y)
        cluster_radius = np.sqrt(np.max((cluster_x - cluster_center_x)**2 + (cluster_y - cluster_center_y)**2))
        
        # Add a circle around the cluster
        circle = Circle((cluster_center_x, cluster_center_y), cluster_radius, color=color, alpha=0.3, fill=True, lw=2)
        ax.add_patch(circle)

    # Remove all borders and axes
    ax.axis('off')

    # Set equal aspect ratio for the plot
    ax.set_aspect('equal', adjustable='datalim')

    # Save the plot with high resolution and no padding
    plt.savefig("clusterATC.png", bbox_inches='tight', pad_inches=0, dpi=600)
    print('save image')
    plt.close()