import pickle
import numpy as np
# import matplotlib.pyplot as plt # Removed
import os

def analyze_results(filename="simulation_results.pkl", output_filename="analysis.txt"):
    """
    Loads simulation results from a pickle file and performs
    a detailed analysis of particle counts, equilibrium, and fluctuations.
    Saves the report to a text file.
    """
    
    if not os.path.exists(filename):
        print(f"Error: Results file not found at '{filename}'")
        print("Please run the simulation script first to generate the results.")
        return

    # --- 1. Load Data ---
    print(f"Loading results from '{filename}'...")
    with open(filename, 'rb') as f:
        results = pickle.load(f)

    snapshots = results['snapshots']
    time_stamps = np.array(results['time_stamps'])
    n = results['lattice_size']
    num_stripes = results['num_stripes']
    trajectories = results['trajectories']
    
    # Handle empty snapshots list (e.g., if simulation time was 0)
    if len(snapshots) == 0 or len(time_stamps) == 0:
        print("Error: No snapshots or timestamps found. Cannot analyze.")
        return

    # --- 2. Process Snapshots ---
    print("Processing snapshots to get particle counts over time...")
    
    # Define stripe boundaries
    stripe_width = n // num_stripes
    stripe_boundaries = []
    for i in range(num_stripes):
        x_start = i * stripe_width
        x_end = x_start + stripe_width
        if i == num_stripes - 1:
            x_end = n # Ensure the last stripe goes to the edge
        stripe_boundaries.append((x_start, x_end))
    
    total_particles_history = []
    # Create a (num_snapshots x num_stripes) array
    stripe_particles_history = np.zeros((len(snapshots), num_stripes))

    for snap_idx, snapshot in enumerate(snapshots):
        # snapshot is a dict {pid: (x, y)}
        total_particles_history.append(len(snapshot))
        
        # Count particles in each stripe for this snapshot
        for x, y in snapshot.values():
            for stripe_idx, (x_start, x_end) in enumerate(stripe_boundaries):
                if x_start <= x < x_end:
                    stripe_particles_history[snap_idx, stripe_idx] += 1
                    break
                    
    total_particles_history = np.array(total_particles_history)

    # --- 3. Calculate Averages ---
    
    # Calculate averages over the *entire* simulation
    avg_total_particles = np.mean(total_particles_history)
    avg_particles_per_stripe = np.mean(stripe_particles_history, axis=0)
    
    # Calculate averages for the *last 25%* of the time (for equilibrium check)
    quarter_index = len(time_stamps) // 4
    if quarter_index > 0:
        avg_total_last_q = np.mean(total_particles_history[-quarter_index:])
        avg_stripe_last_q = np.mean(stripe_particles_history[-quarter_index:, :], axis=0)
    else:
        avg_total_last_q = avg_total_particles
        avg_stripe_last_q = avg_particles_per_stripe


    # --- 4. Find Time to Reach Average ---
    
    def find_time_to_avg(history, avg_val, times):
        rounded_avg = np.round(avg_val)
        # Find all indices where history is >= rounded average
        indices_above_avg = np.where(history >= rounded_avg)[0]
        
        if len(indices_above_avg) > 0:
            first_index = indices_above_avg[0]
            time_to_avg = times[first_index]
            return time_to_avg, first_index
        else:
            # Never reached average
            return np.nan, -1

    time_to_avg_overall, first_index_overall = find_time_to_avg(
        total_particles_history, avg_total_particles, time_stamps
    )
    
    times_to_avg_stripe = []
    first_indices_stripe = []
    for i in range(num_stripes):
        t, idx = find_time_to_avg(
            stripe_particles_history[:, i], avg_particles_per_stripe[i], time_stamps
        )
        times_to_avg_stripe.append(t)
        first_indices_stripe.append(idx)

    # --- 5. Calculate Post-Equilibrium Variance ---
    
    def calc_post_avg_stats(history, first_index):
        if first_index != -1 and first_index < len(history) - 1:
            post_avg_history = history[first_index:]
            variance = np.var(post_avg_history)
            std_dev = np.std(post_avg_history)
            fluctuation_coeff = std_dev / np.mean(post_avg_history) if np.mean(post_avg_history) > 0 else 0
            return variance, std_dev, fluctuation_coeff
        else:
            return np.nan, np.nan, np.nan

    var_total, std_total, fluc_total = calc_post_avg_stats(
        total_particles_history, first_index_overall
    )
    
    vars_stripe = []
    stds_stripe = []
    flucs_stripe = []
    for i in range(num_stripes):
        v, s, f = calc_post_avg_stats(
            stripe_particles_history[:, i], first_indices_stripe[i]
        )
        vars_stripe.append(v)
        stds_stripe.append(s)
        flucs_stripe.append(f)

    # --- 6. Additional Statistics ---
    
    # Particle Density
    total_sites = n * n
    
    avg_density_total = avg_total_particles / total_sites
    avg_density_stripe = [avg_particles_per_stripe[i] / (stripe_boundaries[i][1] - stripe_boundaries[i][0]) / n for i in range(num_stripes)]
    
    avg_density_total_last_q = avg_total_last_q / total_sites
    avg_density_stripe_last_q = [avg_stripe_last_q[i] / (stripe_boundaries[i][1] - stripe_boundaries[i][0]) / n for i in range(num_stripes)]

    # Trajectory Analysis
    total_unique_particles = len(trajectories)
    # -1 because the first position is the 'in' event, not a hop
    hops_per_particle = [len(traj) - 1 for traj in trajectories.values() if len(traj) > 0]
    avg_hops = np.mean(hops_per_particle) if hops_per_particle else 0
    max_hops = np.max(hops_per_particle) if hops_per_particle else 0
    
    # --- 7. Write Report to File ---
    
    print(f"\nWriting analysis report to '{output_filename}'...")
    
    with open(output_filename, 'w') as f:
        f.write("--- KMC Simulation Analysis Report ---\n")
        
        f.write("\n--- Overall Lattice Statistics ---\n")
        f.write(f"Total Simulation Time:        {results['total_time']:.2f}\n")
        f.write(f"Total Lattice Sites:          {total_sites} ({n}x{n})\n")
        f.write(f"Avg. Particle Count (Full):   {avg_total_particles:.2f}\n")
        f.write(f"Avg. Particle Count (Last 25%): {avg_total_last_q:.2f}\n")
        f.write(f"Avg. Density (Full):          {avg_density_total:.4f} particles/site\n")
        f.write(f"Avg. Density (Last 25%):      {avg_density_total_last_q:.4f} particles/site\n")
        f.write(f"Time to Reach Avg. (First):   {time_to_avg_overall:.2f}\n")
        f.write(f"Variance (Post-First Avg.):   {var_total:.2f}\n")
        f.write(f"Std. Dev. (Post-First Avg.):  {std_total:.2f}\n")
        f.write(f"Fluctuation Coeff (Std/Mean): {fluc_total:.4f}\n")

        f.write("\n--- Per-Stripe Statistics ---\n")
        for i in range(num_stripes):
            x_start, x_end = stripe_boundaries[i]
            sites_in_stripe = (x_end - x_start) * n
            f.write(f"\n[Stripe {i} (x = {x_start} to {x_end-1}, {sites_in_stripe} sites)]\n")
            f.write(f"  Avg. Count (Full):            {avg_particles_per_stripe[i]:.2f}\n")
            f.write(f"  Avg. Count (Last 25%):        {avg_stripe_last_q[i]:.2f}\n")
            f.write(f"  Avg. Density (Full):          {avg_density_stripe[i]:.4f} particles/site\n")
            f.write(f"  Avg. Density (Last 25%):      {avg_density_stripe_last_q[i]:.4f} particles/site\n")
            f.write(f"  Time to Reach Avg. (First):   {times_to_avg_stripe[i]:.2f}\n")
            f.write(f"  Variance (Post-First Avg.):   {vars_stripe[i]:.2f}\n")
            f.write(f"  Std. Dev. (Post-First Avg.):  {stds_stripe[i]:.2f}\n")
            f.write(f"  Fluctuation Coeff (Std/Mean): {flucs_stripe[i]:.4f}\n")

        f.write("\n--- Trajectory Statistics ---\n")
        f.write(f"Total Unique Particles Served: {total_unique_particles}\n")
        f.write(f"Avg. Hops per Particle:        {avg_hops:.2f}\n")
        f.write(f"Max. Hops by a Particle:       {max_hops}\n")
    
    print("\nAnalysis complete.")
    print(f"Results saved to '{output_filename}'")


if __name__ == "__main__":
    analyze_results()