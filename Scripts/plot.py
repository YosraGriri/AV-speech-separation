import matplotlib.pyplot as plt

import os


def plot_room(room,
              source_coordinates=None,
              mic_coordinates=None,
              xlim=0,
              ylim=0,
              zlim=0,
              save_path=None):
    """
    Plots a visualization of the provided room, optionally including a source and a microphone array.

    Parameters:
    - room: pyroomacoustics Room object.
    - source_coordinates: List or None, optional. Coordinates of the source in 3D space.
    - mic_coordinates: List or None, optional. Coordinates of microphones in 3D space.
    - xlim, ylim, zlim: float, optional. Limits for the x, y, and z axes in the plot.
    - save_path: str or None, optional. Path to the directory where the plot should be saved. If None, the plot is not saved.
    """
    fig, ax = room.plot()
    if source_coordinates is not None:
        ax.scatter(source_coordinates[0], source_coordinates[1], source_coordinates[2], color='red', marker='o',
                   label='Source')
    if mic_coordinates is not None:
        for i, mic_coord in enumerate(mic_coordinates):
            ax.scatter(mic_coord[0], mic_coord[1], mic_coord[2], color=f'C{i}', marker='^', label=f'Microphone {i + 1}')
        ax.set_xlim([-0.01, xlim + 0.01])
        ax.set_ylim([-0.01, ylim + 0.01])
        ax.set_zlim([-0.01, zlim + 0.01])
    ax.set_title('Room incl. Source and Mic Array')
    ax.legend()

    if save_path is not None:
        save_dir = os.path.join(save_path, 'data', 'shoebox')
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'room_plot.png'))
        print(save_dir)
    plt.show()
