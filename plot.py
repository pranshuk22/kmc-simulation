import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_plots(filename="simulation_results.pkl", output_filename="plots.png"):
    """
    Loads simulation results and generates a multi-panel plot
    saving it to a single image file.
    """
    
    if not os.path.exists(filename):
        print(f"Error: Results file not found at '{filename}'")
        print("Please run the simulation script first to generate the results.")
        return

    print(f"Loading results from '{filename}' for plotting...")
    with open(filename, 'rb') as f:
        results = pickle.load(f)

    snapshots = results['snapshots']
    time_stamps = np.array(results['time_stamps'])
    n = results['lattice_size']
    num_stripes = results['num_stripes']
    
    if len(snapshots) == 0 or len(time_stamps) == 0:
        print("Error: No snapshots or timestamps found. Cannot generate plots.")
        return

    # --- Process Snapshots to get particle counts ---
    print("Processing snapshots for plot data...")
    
    stripe_width = n // num_stripes
    stripe_boundaries = []
    for i in range(num_stripes):
        x_start = i * stripe_width
        x_end = x_start + stripe_width
        if i == num_stripes - 1:
            x_end = n
        stripe_boundaries.append((x_start, x_end))
    
    total_particles_history = []
    stripe_particles_history = np.zeros((len(snapshots), num_stripes))

    for snap_idx, snapshot in enumerate(snapshots):
        total_particles_history.append(len(snapshot))
        for x, y in snapshot.values():
            for stripe_idx, (x_start, x_end) in enumerate(stripe_boundaries):
                if x_start <= x < x_end:
                    stripe_particles_history[snap_idx, stripe_idx] += 1
                    break
                    
    total_particles_history = np.array(total_particles_history)

    # --- Calculate Averages for plotting ---
    avg_total_particles = np.mean(total_particles_history)
    avg_particles_per_stripe = np.mean(stripe_particles_history, axis=0)
    
    quarter_index = len(time_stamps) // 4
    if quarter_index > 0:
        avg_total_last_q = np.mean(total_particles_history[-quarter_index:])
        avg_stripe_last_q = np.mean(stripe_particles_history[-quarter_index:, :], axis=0)
    else:
        avg_total_last_q = avg_total_particles
        avg_stripe_last_q = avg_particles_per_stripe

    def find_time_to_avg(history, avg_val, times):
        rounded_avg = np.round(avg_val)
        indices_above_avg = np.where(history >= rounded_avg)[0]
        if len(indices_above_avg) > 0:
            return times[indices_above_avg[0]]
        return np.nan

    time_to_avg_overall = find_time_to_avg(total_particles_history, avg_total_particles, time_stamps)
    
    # Use post-average data for histogram if equilibrium is reached
    # If not, use the full history
    first_index_overall_for_hist = -1
    if not np.isnan(time_to_avg_overall):
        first_index_overall_for_hist = np.searchsorted(time_stamps, time_to_avg_overall)

    data_to_hist = total_particles_history
    hist_title_suffix = '(Full Simulation)'
    if first_index_overall_for_hist != -1 and first_index_overall_for_hist < len(total_particles_history) - 1:
        data_to_hist = total_particles_history[first_index_overall_for_hist:]
        hist_title_suffix = '(Post-Equilibrium)'

    # --- Generate Plots ---
    print(f"Generating multi-panel plot and saving to '{output_filename}'...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 18)) # 3 rows, 1 column for plots
    fig.suptitle('KMC Simulation Analysis Plots', fontsize=16)

    # Plot 1: Total Particle Count vs. Time
    ax0 = axes[0]
    ax0.plot(time_stamps, total_particles_history, label='Particle Count', alpha=0.8, linewidth=1.5)
    ax0.axhline(avg_total_particles, color='r', linestyle='--', label=f'Overall Avg ({avg_total_particles:.2f})', linewidth=1.5)
    ax0.axhline(avg_total_last_q, color='g', linestyle=':', label=f'Last 25% Avg ({avg_total_last_q:.2f})', linewidth=1.5)
    if not np.isnan(time_to_avg_overall):
        ax0.axvline(time_to_avg_overall, color='purple', linestyle='-.', label=f'Time to Avg ({time_to_avg_overall:.2f})', linewidth=1.5)
    ax0.set_title('1. Total Particle Count vs. Time', fontsize=14)
    ax0.set_xlabel('Time', fontsize=12)
    ax0.set_ylabel('Total Particle Count', fontsize=12)
    ax0.legend(fontsize=10)
    ax0.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax0.set_facecolor('#f9f9f9') # Light background for better contrast

    # Plot 2: Stripe Particle Counts vs. Time
    ax1 = axes[1]
    colors = plt.cm.jet(np.linspace(0, 1, num_stripes))
    for i in range(num_stripes):
        x_start, x_end = stripe_boundaries[i]
        label_text = f'Stripe {i} (x={x_start}-{x_end-1})'
        ax1.plot(time_stamps, stripe_particles_history[:, i], label=label_text, alpha=0.8, color=colors[i], linewidth=1.5)
        ax1.axhline(avg_particles_per_stripe[i], color=colors[i], linestyle=':', alpha=0.7, linewidth=1.0) # Show individual stripe averages
    
    # Optionally, add overall average for comparison
    ax1.axhline(avg_total_particles / num_stripes if num_stripes > 0 else 0, 
                color='black', linestyle='--', label='Global Avg / Num Stripes', alpha=0.7, linewidth=1.5)

    ax1.set_title('2. Particle Counts Per Stripe vs. Time', fontsize=14)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Particle Count', fontsize=12)
    ax1.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5)) # Place legend outside
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.set_facecolor('#f9f9f9')

    # Plot 3: Particle Count Distribution (Histogram)
    ax2 = axes[2]
    # Ensure bins are appropriate even for small data range
    bins = max(1, len(np.unique(data_to_hist)) // 2) if len(np.unique(data_to_hist)) > 1 else 1
    ax2.hist(data_to_hist, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Count Distribution')
    
    if len(data_to_hist) > 0:
        mean_hist_data = np.mean(data_to_hist)
        std_hist_data = np.std(data_to_hist)
        ax2.axvline(mean_hist_data, color='r', linestyle='--', label=f'Mean ({mean_hist_data:.2f})', linewidth=1.5)
        ax2.axvline(mean_hist_data + std_hist_data, color='orange', linestyle=':', label=f'Mean +/- Std ({std_hist_data:.2f})', linewidth=1.0)
        ax2.axvline(mean_hist_data - std_hist_data, color='orange', linestyle=':', linewidth=1.0)

    ax2.set_title(f'3. Distribution of Total Particle Count {hist_title_suffix}', fontsize=14)
    ax2.set_xlabel('Total Particle Count', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_facecolor('#f9f9f9')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent suptitle overlap
    plt.savefig(output_filename, dpi=300)
    print(f"All plots saved to '{output_filename}'")


if __name__ == "__main__":
    generate_plots()