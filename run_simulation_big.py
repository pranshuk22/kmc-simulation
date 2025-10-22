# run_simulation_mega.py

import numpy as np
import pickle
import time
import json  # <-- 1. IMPORT JSON

class Lattice:
    def __init__(self, size):
        self.size = size  # n x n

# ... (MovementMatrixStripes class is unchanged) ...
class MovementMatrixStripes:
    def __init__(self, n, num_stripes, stripe_hop_rates, stripe_diffusion_in, stripe_diffusion_out):
        self.n = n
        self.directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        hop_matrix = np.zeros((n, n, 4))
        diffusion_in_matrix = np.zeros((n, n))
        diffusion_out_matrix = np.zeros((n, n))
        stripe_width = n // num_stripes
        for stripe_index in range(num_stripes):
            rate_hop = stripe_hop_rates[stripe_index]
            rate_in = stripe_diffusion_in[stripe_index]
            rate_out = stripe_diffusion_out[stripe_index]
            x_start = stripe_index * stripe_width
            x_end = x_start + stripe_width
            for x in range(x_start, min(x_end, n)):
                for y in range(n):
                    hop_matrix[x, y, :] = rate_hop
                    diffusion_in_matrix[x, y] = rate_in
                    diffusion_out_matrix[x, y] = rate_out
        self.events = []
        event_rates = []
        for x in range(n):
            for y in range(n):
                for d, (dx, dy) in enumerate(self.directions):
                    rate = float(hop_matrix[x, y, d])
                    if rate > 0:
                        self.events.append(("hop", (x, y), (dx, dy), rate))
                        event_rates.append(rate)
                rate_in = diffusion_in_matrix[x, y]
                if rate_in > 0:
                    self.events.append(("in", (x, y), rate_in))
                    event_rates.append(rate_in)
                rate_out = diffusion_out_matrix[x, y]
                if rate_out > 0:
                    self.events.append(("out", (x, y), rate_out))
                    event_rates.append(rate_out)
        self.cumulative_rates = np.cumsum(event_rates)
        if len(self.cumulative_rates) > 0:
            self.total_rate_val = self.cumulative_rates[-1]
        else:
            self.total_rate_val = 0.0
    @property
    def total_rate(self):
        return self.total_rate_val
    def pick_event(self):
        if self.total_rate_val == 0:
            return None, 0.0
        T = self.total_rate_val
        r = np.random.rand() * T
        index = np.searchsorted(self.cumulative_rates, r, side='right')
        return self.events[index], T


class KMCSimulation:
    # ... (init and _execute are unchanged) ...
    def __init__(self, lattice, movement_matrix, max_unique_particles, total_time, seed,
                 trajectory_writer=None, snapshot_writer=None):
        self.lattice = lattice
        self.mm = movement_matrix
        self.total_time = total_time
        self.max_particles = max_unique_particles
        self.rng = np.random.default_rng(seed)
        self.particles = {}
        self.occupation = {}
        self.available_pids = set(range(self.max_particles))
        self.next_new_pid = self.max_particles
        self.trajectory_writer = trajectory_writer
        self.snapshot_writer = snapshot_writer
    def _execute(self, ev, t):
        kind, data = ev[0], ev[1:]
        if kind == "in":
            site, _ = data
            if site not in self.occupation and len(self.particles) < self.max_particles:
                pid = self.available_pids.pop() if self.available_pids else self.next_new_pid
                if pid == self.next_new_pid: self.next_new_pid += 1
                self.particles[pid] = site
                self.occupation[site] = pid
                if self.trajectory_writer:
                    self.trajectory_writer.write(f"{t:.6f},{pid},in,{site[0]},{site[1]}\n")
                return True
        elif kind == "out":
            site, _ = data
            if site in self.occupation:
                pid = self.occupation[site]
                del self.particles[pid]
                del self.occupation[site]
                if self.trajectory_writer:
                    self.trajectory_writer.write(f"{t:.6f},{pid},out,{site[0]},{site[1]}\n")
                if pid < self.max_particles: self.available_pids.add(pid)
                return True
        elif kind == "hop":
            src, (dx, dy), _ = data
            if src in self.occupation:
                pid_at_src = self.occupation[src]
                dest = ((src[0] + dx) % self.lattice.size, (src[1] + dy) % self.lattice.size)
                if dest not in self.occupation:
                    del self.occupation[src]
                    self.occupation[dest] = pid_at_src
                    self.particles[pid_at_src] = dest
                    if self.trajectory_writer:
                        self.trajectory_writer.write(f"{t:.6f},{pid_at_src},hop,{dest[0]},{dest[1]}\n")
                    return True
        return False
    
    def run(self, max_snapshots=10000):
        t = 0.0
        last_recorded_t = 0.0
        min_interval = self.total_time / max_snapshots
        
        if self.snapshot_writer:
            # --- 2. MODIFIED: Write as JSON ---
            # Old: self.snapshot_writer.write(f"t=0.0\n{str(self.particles)}\n")
            json.dump({'t': 0.0, 'snapshot': self.particles}, self.snapshot_writer)
            self.snapshot_writer.write('\n') # Add a newline
            
        while t < self.total_time:
            ev, total_rate = self.mm.pick_event()
            
            if ev is None or total_rate == 0:
                print("No events possible. Stopping simulation.")
                break
            
            self._execute(ev, t)
            dt = -np.log(self.rng.random()) / total_rate
            t += dt
            
            if t - last_recorded_t >= min_interval:
                if self.snapshot_writer:
                    # --- 3. MODIFIED: Write as JSON ---
                    # Old: self.snapshot_writer.write(f"t={t:.6f}\n{str(self.particles)}\n")
                    json.dump({'t': t, 'snapshot': self.particles}, self.snapshot_writer)
                    self.snapshot_writer.write('\n') # Add a newline
                
                last_recorded_t = t
        return


# ... (Main block is unchanged, but I'll include it for completeness) ...
if __name__ == "__main__":
    # Simulation Parameters
    LATTICE_SIZE = 20
    TOTAL_TIME = 10000
    MAX_SNAPSHOTS = 1000000
    RANDOM_SEED = 42

    # --- Stripe Configuration ---
    NUM_STRIPES = 2
    STRIPE_HOP_RATES =           [1.0, 1.0]
    STRIPE_DIFFUSION_IN_RATES =  [0.2, 0.01]
    STRIPE_DIFFUSION_OUT_RATES = [0.01, 0.2]

    # --- Sanity Checks ---
    assert len(STRIPE_HOP_RATES) == NUM_STRIPES, "Number of hop rates must match NUM_STRIPES."
    assert len(STRIPE_DIFFUSION_IN_RATES) == NUM_STRIPES, "Number of IN-diffusion rates must match NUM_STRIPES."
    assert len(STRIPE_DIFFUSION_OUT_RATES) == NUM_STRIPES, "Number of OUT-diffusion rates must match NUM_STRIPES."
    
    # ------------------------------------------------------------------
    
    start_time = time.time()
    print("Starting KMC simulation (streaming to disk)...")
    
    traj_filename = "trajectories.csv"
    snap_filename = "snapshots.txt" # File is now JSONL
    config_filename = "simulation_config.json"

    with open(traj_filename, "w") as traj_file, open(snap_filename, "w") as snap_file:
        
        traj_file.write("time,pid,event,x,y\n")

        lattice = Lattice(LATTICE_SIZE)
        movement = MovementMatrixStripes(
            n=LATTICE_SIZE,
            num_stripes=NUM_STRIPES,
            stripe_hop_rates=STRIPE_HOP_RATES,
            stripe_diffusion_in=STRIPE_DIFFUSION_IN_RATES,
            stripe_diffusion_out=STRIPE_DIFFUSION_OUT_RATES
        )
        
        sim = KMCSimulation(lattice, movement, LATTICE_SIZE * LATTICE_SIZE, TOTAL_TIME, seed=RANDOM_SEED,
                            trajectory_writer=traj_file,
                            snapshot_writer=snap_file)
        
        sim.run(max_snapshots=MAX_SNAPSHOTS)
    
    config_data = {
        'lattice_size': LATTICE_SIZE,
        'num_stripes': NUM_STRIPES,
        'total_time': TOTAL_TIME,
        'stripe_rates': STRIPE_HOP_RATES,
        'stripe_diffusion_in_rates': STRIPE_DIFFUSION_IN_RATES,
        'stripe_diffusion_out_rates': STRIPE_DIFFUSION_OUT_RATES,
    }
    with open(config_filename, 'w') as f:
        json.dump(config_data, f, indent=4)

    end_time = time.time()
    execution_time = end_time - start_time
        
    print(f"\n Simulation complete...")
    print(f"   - Total execution time: {execution_time:.2f} seconds.")
    print(f"   - Trajectory data saved to: {traj_filename}")
    print(f"   - Snapshot data saved to: {snap_filename} (as JSONL)")
    print(f"   - Config data saved to: {config_filename}\n")