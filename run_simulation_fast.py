# run_simulation_fast.py

import numpy as np
import pickle
import time

class Lattice:
    def __init__(self, size):
        self.size = size  # n x n

class MovementMatrixStripes:
    """
    Creates vertical stripes with user-defined rates for hopping and diffusion.
    
    --- OPTIMIZATION 1 ---
    This class now pre-calculates a cumulative rate array. This allows
    pick_event() to use an O(log M) binary search instead of an O(M)
    linear scan, where M is the total number of events (6 * n * n).
    """
    def __init__(self, n, num_stripes, stripe_hop_rates, stripe_diffusion_in, stripe_diffusion_out):
        self.n = n
        self.directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

        # --- Create Rate Matrices for ALL event types ---
        hop_matrix = np.zeros((n, n, 4))
        diffusion_in_matrix = np.zeros((n, n))
        diffusion_out_matrix = np.zeros((n, n))
        
        stripe_width = n // num_stripes

        # Populate all rate matrices based on stripe index
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
        
        # --- Build the fast event catalog ---
        # self.events is now a flat list of all possible events
        # self.cumulative_rates is a parallel list for binary search
        self.events = []
        event_rates = []

        for x in range(n):
            for y in range(n):
                # Hopping events
                for d, (dx, dy) in enumerate(self.directions):
                    rate = float(hop_matrix[x, y, d])
                    if rate > 0:
                        self.events.append(("hop", (x, y), (dx, dy), rate))
                        event_rates.append(rate)
                
                # Diffusion events
                rate_in = diffusion_in_matrix[x, y]
                if rate_in > 0:
                    self.events.append(("in", (x, y), rate_in))
                    event_rates.append(rate_in)
                
                rate_out = diffusion_out_matrix[x, y]
                if rate_out > 0:
                    self.events.append(("out", (x, y), rate_out))
                    event_rates.append(rate_out)
        
        # Create the cumulative rate array and store the total rate
        self.cumulative_rates = np.cumsum(event_rates)
        if len(self.cumulative_rates) > 0:
            self.total_rate_val = self.cumulative_rates[-1]
        else:
            self.total_rate_val = 0.0

    @property
    def total_rate(self):
        # Now just a fast lookup
        return self.total_rate_val

    def pick_event(self):
        """
        Picks an event using O(log M) binary search.
        """
        if self.total_rate_val == 0:
            return None, 0.0

        T = self.total_rate_val
        r = np.random.rand() * T
        
        # Use binary search to find the index
        index = np.searchsorted(self.cumulative_rates, r, side='right')
        
        return self.events[index], T

class KMCSimulation:
    """
    --- OPTIMIZATION 2 ---
    This class now uses two dictionaries for O(1) lookups:
    1. self.particles:  pid -> site
    2. self.occupation: site -> pid
    This avoids O(P) loops in the _execute method.
    """
    def __init__(self, lattice, movement_matrix, max_unique_particles, total_time, seed):
        self.lattice = lattice
        self.mm = movement_matrix
        self.total_time = total_time
        self.max_particles = max_unique_particles
        self.rng = np.random.default_rng(seed)
        
        # self.particles: {pid: (x, y)}
        self.particles = {}
        # self.occupation: {(x, y): pid} - for O(1) lookups
        self.occupation = {}
        
        self.available_pids = set(range(self.max_particles))
        self.next_new_pid = self.max_particles
        self.time_stamps = [0.0]
        self.snapshots = [dict()]
        self.trajectories = {}

    def _execute(self, ev):
        kind, data = ev[0], ev[1:]
        
        if kind == "in":
            site, _ = data
            # O(1) check instead of O(P)
            if site not in self.occupation and len(self.particles) < self.max_particles:
                pid = self.available_pids.pop() if self.available_pids else self.next_new_pid
                if pid == self.next_new_pid: self.next_new_pid += 1
                
                # Update both maps
                self.particles[pid] = site
                self.occupation[site] = pid
                
                self.trajectories.setdefault(pid, []).append(site)
                return True
                
        elif kind == "out":
            site, _ = data
            # O(1) check and lookup
            if site in self.occupation:
                pid = self.occupation[site]
                
                # Update both maps
                del self.particles[pid]
                del self.occupation[site]
                
                if pid < self.max_particles: self.available_pids.add(pid)
                return True
                
        elif kind == "hop":
            src, (dx, dy), _ = data
            # O(1) check and lookup
            if src in self.occupation:
                pid_at_src = self.occupation[src]
                dest = ((src[0] + dx) % self.lattice.size, (src[1] + dy) % self.lattice.size)
                
                # O(1) check
                if dest not in self.occupation:
                    # Update both maps
                    del self.occupation[src]
                    self.occupation[dest] = pid_at_src
                    self.particles[pid_at_src] = dest
                    
                    self.trajectories[pid_at_src].append(dest)
                    return True
        return False
    
    def run(self, max_snapshots=10000):
        t = 0.0
        last_recorded_t = 0.0
        min_interval = self.total_time / max_snapshots
        
        while t < self.total_time:
            ev, total_rate = self.mm.pick_event()
            
            if ev is None or total_rate == 0:
                print("No events possible. Stopping simulation.")
                break
                
            self._execute(ev)
            
            dt = -np.log(self.rng.random()) / total_rate
            t += dt
            
            if t - last_recorded_t >= min_interval:
                # We copy to avoid snapshots changing later
                self.snapshots.append(dict(self.particles))
                self.time_stamps.append(t)
                last_recorded_t = t
        
        return self.trajectories, self.snapshots, self.time_stamps


# Main execution block
if __name__ == "__main__":
    # Simulation Parameters
    LATTICE_SIZE = 20
    TOTAL_TIME = 1000
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
    print("Starting KMC simulation (Fast Algorithmic Version)...")
    
    lattice = Lattice(LATTICE_SIZE)
    movement = MovementMatrixStripes(
        n=LATTICE_SIZE,
        num_stripes=NUM_STRIPES,
        stripe_hop_rates=STRIPE_HOP_RATES,
        stripe_diffusion_in=STRIPE_DIFFUSION_IN_RATES,
        stripe_diffusion_out=STRIPE_DIFFUSION_OUT_RATES
    )
    sim = KMCSimulation(lattice, movement, LATTICE_SIZE * LATTICE_SIZE, TOTAL_TIME, seed=RANDOM_SEED)
    
    trajectories, snapshots, time_stamps = sim.run()
    
    results = {
        'lattice_size': LATTICE_SIZE,
        'num_stripes': NUM_STRIPES,
        'total_time': TOTAL_TIME,
        'trajectories': trajectories,
        'snapshots': snapshots,
        'time_stamps': time_stamps,
        'stripe_rates': STRIPE_HOP_RATES,
        'stripe_diffusion_in_rates': STRIPE_DIFFUSION_IN_RATES,
        'stripe_diffusion_out_rates': STRIPE_DIFFUSION_OUT_RATES,
    }
    
    output_filename = "simulation_results.pkl"
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)

    end_time = time.time()
    execution_time = end_time - start_time
        
    print(f"\n Simulation complete...")
    print(f"   - Total execution time: {execution_time:.2f} seconds.")