import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_room(room, source_coordinates=None, mic_coordinates=None, xlim=0, ylim=0, zlim=0, save_path=None):

    """
    Visualizes a 3D room with sources and microphone coordinates.

    Parameters:
    - room: Room object that contains the room configuration and has a method to plot itself.
    - source_coordinates: List of tuples/lists, optional. Each element is the (x, y, z) coordinates of a source.
    - mic_coordinates: List of tuples/lists, optional. Each element is the (x, y, z) coordinates of a microphone.
    - xlim: float, optional. The limit for the x-axis of the plot.
    - ylim: float, optional. The limit for the y-axis of the plot.
    - zlim: float, optional. The limit for the z-axis of the plot.
    - save_path: str or pathlib.Path, optional. The directory where the plot should be saved as a PDF. If None, the plot is not saved.

    Returns:
    - None. The function creates a 3D plot of the room and displays it. If save_path is provided, the plot is saved as a PDF in the specified directory.
    """

    # Colors from the new palette
    wall_colors = ['#eaf7f8', '#f1e8e6', '#e9dccc', '#e8c2b1', '#b6957c']
    source_color = '#22577a'
    mic_color = '#31572c'

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    room.plot(fig=fig, ax=ax)  # Assuming room has a method to plot itself

    # Draw colored walls
    walls = [
        [[0, 0, 0], [0, ylim, 0], [0, ylim, zlim], [0, 0, zlim]],  # x=0 wall
        [[xlim, 0, 0], [xlim, ylim, 0], [xlim, ylim, zlim], [xlim, 0, zlim]],  # x=xlim wall
        [[0, 0, 0], [xlim, 0, 0], [xlim, 0, zlim], [0, 0, zlim]],  # y=0 wall
        [[0, ylim, 0], [xlim, ylim, 0], [xlim, ylim, zlim], [0, ylim, zlim]],  # y=ylim wall
        [[0, 0, 0], [xlim, 0, 0], [xlim, ylim, 0], [0, ylim, 0]],  # z=0 floor
        [[0, 0, zlim], [xlim, 0, zlim], [xlim, ylim, zlim], [0, ylim, zlim]],  # z=zlim ceiling
    ]

    for i, wall in enumerate(walls):
        ax.add_collection3d(Poly3DCollection([wall], color=wall_colors[i % len(wall_colors)], alpha=0.1))

    # Plotting sources with different symbols and transparency
    source_markers = ['x']
    if source_coordinates:
        for i, source_coord in enumerate(source_coordinates):
            ax.scatter(*source_coord, color=source_color,
                       marker=source_markers[i % len(source_markers)], s=80,
                       label=f'Source {i + 1}', alpha=1)

    # Plotting microphones
    if mic_coordinates:
        for mic_coord in mic_coordinates:
            ax.scatter(*mic_coord, color=mic_color, marker='o', s=50,
                       label='Microphones', alpha=1)

    # Setting axis limits
    ax.set_xlim([0, xlim])
    ax.set_ylim([0, ylim])
    ax.set_zlim([0, zlim])

    # Customizing the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.05, 1),
              fontsize='small', frameon=True, facecolor='white', framealpha=0.7)
    ax.grid(True)  # Adding a grid for better visualization
    ax.set_box_aspect([1, 1, 1])  # Keeping the aspect ratio square

    # Check if save path is provided and save the plot as a PDF
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'room_plot.pdf'), format='pdf')
        print(f"Plot saved as PDF in: {save_path}")
    plt.show()
