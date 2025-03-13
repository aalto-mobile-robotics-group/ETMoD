import numpy as np
import matplotlib.pyplot as plt

def Draw_cluster_results(data,centers,membership,w):
    # velocity = data.iloc[:, 4].to_numpy()  
    # motion_angle = data.iloc[:, 5].to_numpy()
    velocity = data.iloc[:, 0].to_numpy()  
    motion_angle = data.iloc[:, 1].to_numpy()
    if centers is not None:
        if w == None:
            center_angles = centers[:,1]
        else :
            center_angles = np.arctan2(centers[:, 2]/w, centers[:, 1]/w)  # Motion angle of centers
        center_radii = centers[:, 0]  # Velocity of centers (first column)

    # Unique cluster labels
    unique_clusters = np.unique(membership)

    # Define colors for the clusters
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    # Plot in polar coordinates
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # Plot data points for each cluster
    for cluster, color in zip(unique_clusters, colors):
        mask = membership == cluster
        ax.scatter(motion_angle[mask], velocity[mask], 
                c=[color], alpha=0.7, s=10, label=f"Cluster {cluster+1}")

    # Plot cluster centers
    if centers is not None:
        ax.scatter(center_angles, center_radii, 
                c='black', s=100, marker='x', label="Cluster Centers")

    # Add title and legend
    ax.set_title("Clustering Results in Polar Coordinates", va='bottom')
    ax.legend()

    # Show the plot
    plt.savefig("cluster_vel_X57", dpi=300, bbox_inches='tight')