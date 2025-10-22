# run_simulation.py

import numpy as np
import pickle
import time

class Lattice:
    def __init__(self, size):
        self.size = size  # n x n

class MovementMatrixStripes:
    """
    Creates vertical stripes with user-defined rates for hopping and diffusion.
    """
    def __init__(self, n, num_stripes, stripe_hop_rates, stripe_diffusion_in, stripe_diffusion_out):
        self.n = n
        self.directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

        # --- Create Rate Matrices for ALL event types ---
        hop_matrix = np.zeros((n, n, 4))
        self.diffusion_in_matrix = np.zeros((n, n))  # <-- NEW: 2D matrix for IN rates
        self.diffusion_out_matrix = np.zeros((n, n)) # <-- NEW: 2D matrix for OUT rates
        
        stripe_width = n // num_stripes

        # Populate all rate matrices based on stripe index
        for stripe_index in range(num_stripes):
            # Get the rates for this specific stripe
            rate_hop = stripe_hop_rates[stripe_index]
            rate_in = stripe_diffusion_in[stripe_index]
            rate_out = stripe_diffusion_out[stripe_index]
            
            x_start = stripe_index * stripe_width
            x_end = x_start + stripe_width
            
            for x in range(x_start, min(x_end, n)):
                for y in range(n):
                    hop_matrix[x, y, :] = rate_hop
                    self.diffusion_in_matrix[x, y] = rate_in
                    self.diffusion_out_matrix[x, y] = rate_out

        self.hop_rate_matrix = hop_matrix
        self.events = self._build_events()

    def _build_events(self):
        """Builds event list using the site-specific rate matrices."""
        events = []
        for x in range(self.n):
            for y in range(self.n):
                # Hopping events
                for d, (dx, dy) in enumerate(self.directions):
                    rate = float(self.hop_rate_matrix[x, y, d])
                    events.append(("hop", (x, y), (dx, dy), rate))
                
                # Diffusion events with site-specific rates
                rate_in = self.diffusion_in_matrix[x, y]
                rate_out = self.diffusion_out_matrix[x, y]
                events.append(("in", (x, y), rate_in))
                events.append(("out", (x, y), rate_out))
        return events

    @property
    def total_rate(self):
        return sum(ev[-1] for ev in self.events)

    def pick_event(self):
        T = self.total_rate
        r = np.random.rand() * T
        cum = 0.0
        for ev in self.events:
            cum += ev[-1]
            if r < cum:
                return ev, T
        return self.events[-1], T

class KMCSimulation:
    # ... (No changes needed in this class) ...
    def __init__(self, lattice, movement_matrix, max_unique_particles, total_time, seed):
        self.lattice = lattice
        self.mm = movement_matrix
        self.total_time = total_time
        self.max_particles = max_unique_particles
        self.rng = np.random.default_rng(seed)
        
        self.particles = {}
        self.available_pids = set(range(self.max_particles))
        self.next_new_pid = self.max_particles
        self.time_stamps = [0.0]
        self.snapshots = [dict()]
        self.trajectories = {}

    def _occupied(self):
        return set(self.particles.values())
    
    def _execute(self, ev):
        kind, data = ev[0], ev[1:]
        if kind == "in":
            site, _ = data
            if site not in self._occupied() and len(self.particles) < self.max_particles:
                pid = self.available_pids.pop() if self.available_pids else self.next_new_pid
                if pid == self.next_new_pid: self.next_new_pid += 1
                self.particles[pid] = site
                self.trajectories.setdefault(pid, []).append(site)
                return True
        elif kind == "out":
            site, _ = data
            for pid, pos in list(self.particles.items()):
                if pos == site:
                    del self.particles[pid]
                    if pid < self.max_particles: self.available_pids.add(pid)
                    return True
        elif kind == "hop":
            src, (dx, dy), _ = data
            pid_at_src = next((pid for pid, pos in self.particles.items() if pos == src), None)
            if pid_at_src is not None:
                dest = ((src[0] + dx) % self.lattice.size, (src[1] + dy) % self.lattice.size)
                if dest not in self._occupied():
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
            self._execute(ev)
            
            dt = -np.log(self.rng.random()) / total_rate
            t += dt
            
            if t - last_recorded_t >= min_interval:
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
    # The number of items in EACH list below MUST match NUM_STRIPES.
    STRIPE_HOP_RATES =           [1.0, 1.0]   # Hopping speed in each stripe
    STRIPE_DIFFUSION_IN_RATES =  [0.2, 0.01]   # Particle appearance rate
    STRIPE_DIFFUSION_OUT_RATES = [0.01, 0.2]   # Particle disappearance rate

    # --- Sanity Checks ---
    assert len(STRIPE_HOP_RATES) == NUM_STRIPES, "Number of hop rates must match NUM_STRIPES."
    assert len(STRIPE_DIFFUSION_IN_RATES) == NUM_STRIPES, "Number of IN-diffusion rates must match NUM_STRIPES."
    assert len(STRIPE_DIFFUSION_OUT_RATES) == NUM_STRIPES, "Number of OUT-diffusion rates must match NUM_STRIPES."
    
    # ------------------------------------------------------------------
    
    start_time = time.time()
    print("Starting KMC simulation...")
    
    lattice = Lattice(LATTICE_SIZE)
    # Pass all the rate lists to the constructor
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
        'stripe_diffusion_in_rates': STRIPE_DIFFUSION_IN_RATES,  # Save for potential use
        'stripe_diffusion_out_rates': STRIPE_DIFFUSION_OUT_RATES, # Save for potential use
    }
    
    output_filename = "simulation_results.pkl"
    with open(output_filename, 'wb') as f:
        pickle.dump(results, f)

    end_time = time.time()
    execution_time = end_time - start_time
        
    print(f"\n Simulation complete...")
    print(f"   - Total execution time: {execution_time:.2f} seconds.")