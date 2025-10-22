# visualize.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pickle
import sys
import os

class LatticePlot:
    """
    Visualizer for the KMC simulation that shows particle numbers
    at their current positions, without displaying their historical paths.
    """
    def __init__(self, data):
        # Load all data from the results dictionary
        self.lattice_size = data['lattice_size']
        self.num_stripes = data['num_stripes']
        self.snapshots = data['snapshots']
        self.time_stamps = data['time_stamps']
        self.stripe_rates = data.get('stripe_rates', [])
        
        # Setup the plot
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.suptitle("Interactive Kinetic Monte Carlo Simulation", fontsize=16, weight='bold')
        plt.subplots_adjust(bottom=0.2, top=0.85)
        
        self._configure_axes()
        self._draw_background_stripes()
        
        # This list will hold the text artists for particle numbers
        self.particle_texts = []
        
        # Info box for time and particle count
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      fontsize=12, family='monospace', va='top', ha='left',
                                      bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.8))
        
        self._create_slider()

    def _configure_axes(self):
        """Set up titles, limits, and grid for the main plot axis."""
        self.ax.set_xlabel("X-coordinate")
        self.ax.set_ylabel("Y-coordinate")
        self.ax.set_xlim(-0.5, self.lattice_size - 0.5)
        self.ax.set_ylim(-0.5, self.lattice_size - 0.5)
        self.ax.set_aspect('equal')

    def _draw_background_stripes(self):
        """Draw colored vertical rectangles to indicate high/low rate regions."""
        stripe_width = self.lattice_size / self.num_stripes
        stripe_colors = ['#d6e8ee', '#f5dcdc']  # Light blue and light red
        for i in range(self.num_stripes):
            rate = self.stripe_rates[i]
            rate_text = f"Fast ({rate})" if rate > 0.1 else f"Slow ({rate})"
            x_start = i * stripe_width
            self.ax.add_patch(
                plt.Rectangle((x_start - 0.5, -0.5), stripe_width, self.lattice_size,
                              color=stripe_colors[i % 2], alpha=0.5, zorder=0)
            )
            self.ax.text(x_start + stripe_width/2 - 0.5, self.lattice_size + 0.5,
                         rate_text, ha='center', va='bottom', fontsize=9)
            if i > 0:
                self.ax.axvline(x=x_start - 0.5, color='gray', linestyle='--')

    def _create_slider(self):
        """Create and configure the time slider widget."""
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(
            ax=ax_slider,
            label='Time (s)',
            valmin=0.0,
            valmax=self.time_stamps[-1],
            valinit=0.0,
            valfmt='%.2f s'
        )
        self.slider.on_changed(self.update)

    def update(self, val):
        """Function called when the slider value changes."""
        time = self.slider.val
        step_idx = np.searchsorted(self.time_stamps, time, side="right") - 1
        current_snapshot = self.snapshots[step_idx]

        # 1. Clear the old numbers from the plot
        for txt in self.particle_texts:
            txt.remove()
        self.particle_texts.clear()

        # 2. Draw new numbers for the current frame
        for pid, (x, y) in current_snapshot.items():
            txt = self.ax.text(x, y, str(pid), ha='center', va='center',
                               color='black', fontsize=8, weight='bold', zorder=3,
                               bbox=dict(boxstyle='circle,pad=0.1', fc='white', ec='none', alpha=0.6))
            self.particle_texts.append(txt)
        
        # 3. Update info text box
        self.info_text.set_text(f"Time: {time:.2f} s\nParticles: {len(current_snapshot)}")
        
        self.fig.canvas.draw_idle()

    def show(self):
        self.update(0)  # Initialize plot at t=0
        plt.show()

def plot_particle_count(data):
    """
    Generates two separate plots:
    1. A plot of the total particle count vs. time.
    2. If num_stripes > 1, a second plot with a subplot for each stripe.
    """
    # --- Extract data from the dictionary ---
    snapshots = data['snapshots']
    time_stamps = data['time_stamps']
    num_stripes = data['num_stripes']
    lattice_size = data['lattice_size']
    
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Total System Occupancy (Always generated) ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    total_particle_counts = [len(snap) for snap in snapshots]
    avg_count = np.mean(total_particle_counts)
    
    ax1.plot(time_stamps, total_particle_counts, lw=2, color="teal", label="Total Particle Count")
    ax1.axhline(avg_count, color="crimson", linestyle="--", lw=2, label=f"Avg = {avg_count:.2f}")
    
    ax1.set_title("Total System Occupancy Over Time", fontsize=16)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Number of Particles")
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Occupancy per Stripe (Generated only if num_stripes > 1) ---
    if num_stripes > 1:
        # Calculate stripe boundaries
        stripe_width = lattice_size // num_stripes
        
        # Pre-calculate counts for each stripe for each snapshot
        stripe_counts = [[] for _ in range(num_stripes)]
        for snap in snapshots:
            counts_in_snapshot = [0] * num_stripes
            for x, y in snap.values():
                stripe_index = min(x // stripe_width, num_stripes - 1)
                counts_in_snapshot[stripe_index] += 1
            
            for i in range(num_stripes):
                stripe_counts[i].append(counts_in_snapshot[i])

        # Create subplots
        fig2, axes2 = plt.subplots(
            nrows=num_stripes, 
            ncols=1, 
            figsize=(10, 2.5 * num_stripes), 
            sharex=True
        )
        fig2.suptitle("Particle Occupancy per Stripe", fontsize=16, y=0.95)

        for i in range(num_stripes):
            ax = axes2[i]
            counts = stripe_counts[i]
            avg_count_stripe = np.mean(counts)
            
            ax.plot(time_stamps, counts, lw=2, label=f"Stripe {i+1} Count")
            ax.axhline(avg_count_stripe, color="crimson", linestyle="--", lw=2, label=f"Avg = {avg_count_stripe:.2f}")
            ax.set_ylabel("Particle Count")
            ax.legend(loc='upper right')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        axes2[-1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()


if __name__ == "__main__":
    INPUT_FILENAME = "simulation_results.pkl"
    
    if not os.path.exists(INPUT_FILENAME):
        print(f"❌ Error: Results file '{INPUT_FILENAME}' not found.")
        print("➡️ Please run 'python run_simulation.py' first to generate the data.")
        sys.exit(1)
        
    with open(INPUT_FILENAME, 'rb') as f:
        simulation_data = pickle.load(f)
    print("✅ Data loaded successfully. Launching visualizer...")

    # Launch the interactive plot
    plot = LatticePlot(simulation_data)
    plot.show()
    
    # Launch the particle count plots
    plot_particle_count(simulation_data)