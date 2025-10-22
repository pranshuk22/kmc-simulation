# animate.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import sys
import os

# --- ⚙️ Animation Parameters (Modify these) ---
FPS = 20  # Frames per second for the output video
DPI = 200 # Dots per inch (resolution) for the output video
INPUT_FILENAME = "simulation_results.pkl"
OUTPUT_FILENAME = "simulation_video.mp4"

# --- 1. Load Simulation Data ---
if not os.path.exists(INPUT_FILENAME):
    print(f"Error: Results file '{INPUT_FILENAME}' not found.")
    print("Please run 'python run_simulation.py' first to generate the data.")
    sys.exit(1)

print(f"Loading simulation data from '{INPUT_FILENAME}'...")
with open(INPUT_FILENAME, 'rb') as f:
    data = pickle.load(f)

lattice_size = data['lattice_size']
num_stripes = data['num_stripes']
snapshots = data['snapshots']
time_stamps = data['time_stamps']
stripe_rates = data.get('stripe_rates', [])
print("Data loaded successfully.")

# --- 2. Set up the Plot Figure ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(8, 6))

# Configure axes properties
ax.set_xlim(-0.5, lattice_size - 0.5)
ax.set_ylim(-0.5, lattice_size - 0.5)
ax.set_xlabel("X-coordinate")
ax.set_ylabel("Y-coordinate")
ax.set_aspect('equal')

# Draw the static background stripes
stripe_width = lattice_size / num_stripes
stripe_colors = ['#d6e8ee', '#f5dcdc']
for i in range(num_stripes):
    rate = stripe_rates[i]
    x_start = i * stripe_width
    ax.add_patch(
        plt.Rectangle((x_start - 0.5, -0.5), stripe_width, lattice_size,
                      color=stripe_colors[i % 2], alpha=0.5, zorder=0)
    )
    ax.text(x_start + stripe_width/2 - 0.5, lattice_size + 0.5, f"Rate: {rate}",
             ha='center', va='bottom', fontsize=9)

# A list to hold the text artists for particle numbers
particle_texts = []
# Dynamic title to show current time
time_text = ax.set_title('')

# --- 3. Define the Animation Update Function ---
# This function is called for each frame of the video
def update(frame):
    """
    Clears the previous frame's particles and draws the new ones.
    """
    current_snapshot = snapshots[frame]
    
    # Clear the numbers from the previous frame
    for txt in particle_texts:
        txt.remove()
    particle_texts.clear()

    # Draw the new particle numbers for the current frame
    for pid, (x, y) in current_snapshot.items():
        txt = ax.text(x, y, str(pid), ha='center', va='center',
                      color='black', fontsize=8, weight='bold', zorder=3,
                      bbox=dict(boxstyle='circle,pad=0.1', fc='white', ec='none', alpha=0.7))
        particle_texts.append(txt)
    
    # Update the title with the current simulation time
    current_time = time_stamps[frame]
    time_text.set_text(f'Simulation Time: {current_time:.2f} s')
    
    # Return the artists that were modified
    return particle_texts + [time_text]

# --- 4. Create and Save the Animation ---
# Create the animation object
# blit=False is more robust for saving animations with changing text elements
anim = animation.FuncAnimation(
    fig,
    update,
    frames=len(snapshots),
    interval=1000/FPS,
    blit=False
)

# Save the animation to a file
print(f"\n Generating video... this may take a few moments.")
anim.save(OUTPUT_FILENAME, writer='ffmpeg', fps=FPS, dpi=DPI)

print(f"\n Success! Animation saved as '{OUTPUT_FILENAME}'")