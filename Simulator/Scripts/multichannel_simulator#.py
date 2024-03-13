import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pra
import math

# Define Room Dimensions
room_dimensions = [4, 4, 2.5]  # Length, Width, Height in meters

# Create Room
shoebox = pra.ShoeBox(room_dimensions, fs=16000, max_order=3)

# Step 3: Visualizing the Empty Room
fig, ax = shoebox.plot()
ax.set_title("Empty Room")
plt.show()

# Height of a person (170 cm)
person_height = 1.7
center_x, center_y = room_dimensions[0]/2, room_dimensions[1]/2

# Distance from the center to each microphone (assume glasses width of 20 cm)
glasses_width = 0.139
mic_array = np.array([
    [center_x + glasses_width/2, center_y + glasses_width/2, person_height],
    [center_x + glasses_width/2, center_y - glasses_width/2, person_height],
    [center_x - glasses_width/2, center_y + glasses_width/2, person_height],
    [center_x - glasses_width/2, center_y - glasses_width/2, person_height]
])

# Print information about room dimensions
print("Room dimensions array shape:", np.array(room_dimensions)[:2])
print("Room dimensions array:", np.array(room_dimensions))

# Calculate microphone positions
mic_positions = mic_array

# Create a list of colors for each microphone
colors = ['r', 'g', 'b', 'c']

# Add Microphone Array to the shoebox
shoebox.add_microphone_array(pra.MicrophoneArray(mic_positions.T, shoebox.fs))

# Visualize the Room with Microphones
fig, ax = shoebox.plot()
ax.set_title("Room with Microphones")

# Plot each microphone with different colors
for i, mic_pos in enumerate(mic_positions):
    ax.scatter(mic_pos[0], mic_pos[1], mic_pos[2], color=colors[i], s=100, label=f'Microphone {i+1}')

plt.show()

# Speaker information
speaker_height = 1.8
speaker_position = [center_x + math.sqrt(2)/2, center_y - math.sqrt(2)/2, speaker_height]
print(f"Speaker Position: X = {speaker_position[0]}, Y = {speaker_position[1]}, Z = {speaker_position[2]}")
# Adding source
shoebox.add_source(speaker_position)

# Visualize the Room with Microphones and Speaker
ax.set_title("Room with Microphones and Speaker")

# Plot each microphone with different colors
for i, mic_pos in enumerate(mic_positions):
    ax.scatter(mic_pos[0], mic_pos[1], mic_pos[2], color=colors[i], s=100, label=f'Microphone {i+1}')
ax.scatter(speaker_position[0], speaker_position[1], speaker_position[2], color='m', s=100, label='Speaker')

plt.show()
