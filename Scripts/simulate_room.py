import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
import math


def add_microphone_array(
        room,
        array_location):
    """
    Adds a microphone array to the specified room at the specified location.

    Parameters:
    - room: pyroomacoustics Room object.
    - array_location: 2D numpy array or similar, representing the coordinates of microphones in 3D space.
    """
    array = pra.MicrophoneArray(array_location.T, room.fs)
    room.add_microphone_array(array)


def add_sound_source(
        room,
        source_location,
        signal):
    """
    Adds a sound source to the specified room at the specified location.

    Parameters:
    - room: pyroomacoustics Room object.
    - source_location: List or numpy array, representing the coordinates of the sound source in 3D space.
    """

    room.add_source(source_location, signal)



def create_room(fs,
                shoebox=True,
                room_dimensions=None,
                room_corners=None,
                absorption_coefficient=0.2,
                max_order=1,
                materials=pra.Material(0.5, 0.15),
                ray_tracing=True,
                air_absorption=True):
    """
    Simulates a room with reverberation using pyroomacoustics.

    Parameters:
    - fs: int, Sampling frequency.
    - shoebox: bool, Whether to use the shoebox model (True) or corners model (False).
    - room_dimensions: tuple or None, Dimensions of the room (length, width, height) for the shoebox model.
    - room_corners: numpy array or None, Coordinates of the corners for the corners model.
    - absorption_coefficient: float, Absorption coefficient for the room's walls (0.0 to 1.0).
    - max_order: int, Maximum order of reflections for the shoebox model.
    - materials: pyroomacoustics Material object, Acoustic material properties for the room's surfaces.
    - ray_tracing: bool, Whether to use ray tracing for sound propagation modeling.
    - air_absorption: bool, Whether to consider air absorption in the simulation.

    Returns:
    - room: pyroomacoustics Room object representing the simulated room.
    """
    if shoebox:
        if room_dimensions is None:
            raise ValueError("Please provide room dimensions for the shoebox model.")
        room = pra.ShoeBox(room_dimensions, fs=fs, max_order=max_order, materials=materials,
                           ray_tracing=ray_tracing, air_absorption=air_absorption)
    else:
        if room_corners is None:
            raise ValueError("Please provide room corners for the corners model.")
        room = pra.Room.from_corners(room_corners)
    return room
