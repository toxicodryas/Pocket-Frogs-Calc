import numpy as np
from numba import jit
import time

base_colors = [
    "Maroon", "Red", "Tangelo", "Orange", "Golden", "Yellow", "Lime", "Green", "Emerald", "Olive", "Marine",
    "Aqua", "Azure", "Blue", "Purple", "Royal", "Violet", "Pink", "Beige", "Cocos", "Black", "White", "Glass"
]

secondary_colors = [
    "Tingo", "Carota", "Aurum", "Folium", "Muscus", "Callaina", "Caelus", "Pruni", "Viola", "Floris", "Ceres",
    "Albeo", "Bruna", "Cafea", "Picea", "Chroma"
]

color_wheel = [
    ("Maroon", "Tingo"),
    ("Red", "Tingo"),
    ("Tangelo", "Carota"),
    ("Orange", "Carota"),
    ("Golden", "Aurum"),
    ("Yellow", "Aurum"),
    ("Lime", "Folium"),
    ("Green", "Folium"),
    ("Emerald", "Muscus"),
    ("Olive", "Muscus"),
    ("Marine", "Callaina"),
    ("Aqua", "Callaina"),
    ("Azure", "Caelus"),
    ("Blue", "Caelus"),
    ("Purple", "Pruni"),
    ("Royal", "Viola"),
    ("Pink", "Ceres"),
    ("Violet", "Floris"),
    ("Beige", "Bruna"),
    ("Cocos", "Cafea"),
    ("Black", "Picea"),
    ("White", "Albeo"),
    ("Glass", "Chroma")
]

# Create lookup dictionaries for O(1) index mapping
base_to_idx = {color: i for i, color in enumerate(base_colors)}
secondary_to_idx = {color: i for i, color in enumerate(secondary_colors)}

# Convert color wheel to indices for numba functions
color_wheel_indices = np.array([
    (base_to_idx[base], secondary_to_idx[secondary]) 
    for base, secondary in color_wheel
], dtype=np.int32)

@jit(nopython=True)
def breed_pair_numba(frog_1_base_idx, frog_1_sec_idx, frog_2_base_idx, frog_2_sec_idx):
    """Breed two frogs and return offspring indices"""
    base_idx = frog_1_base_idx if np.random.random() < 0.5 else frog_2_base_idx
    sec_idx = frog_1_sec_idx if np.random.random() < 0.5 else frog_2_sec_idx
    return base_idx, sec_idx

@jit(nopython=True)
def strategy_1_numba(frog_table, color_wheel_indices):
    """Strategy 1 implemented with numba for speed"""
    total_breeds = 0
    n_frogs = len(color_wheel_indices)
    
    # Loop across every frog in the color wheel
    for i in range(n_frogs):
        base_breeds = 0
        frog_1_base, frog_1_sec = color_wheel_indices[i]
        
        # And breed with every other frog in the wheel ahead of it
        for j in range(i + 1, n_frogs):
            num_breeds = 0
            frog_2_base, frog_2_sec = color_wheel_indices[j]
            
            # Check if the 2 unique offspring are in the table
            while not (frog_table[frog_1_base, frog_2_sec] and frog_table[frog_2_base, frog_1_sec]):
                # Breed frogs, and update table 
                offspring_base, offspring_secondary = breed_pair_numba(
                    frog_1_base, frog_1_sec, frog_2_base, frog_2_sec
                )
                frog_table[offspring_base, offspring_secondary] = True
                num_breeds += 1

            base_breeds += num_breeds

        total_breeds += base_breeds

    return total_breeds

@jit(nopython=True)
def strategy_2_numba(frog_table, color_wheel_indices):
    """Strategy 2 implemented with numba for speed"""
    total_breeds = 0
    n_frogs = len(color_wheel_indices)
    
    # Loop across every frog in the color wheel
    for i in range(n_frogs):
        base_breeds = 0
        frog_1_base, frog_1_sec = color_wheel_indices[i]
        
        # Check if the secondary color is redundant 
        secondary_is_redundant = False
        if i < n_frogs - 1:  # Not the last frog
            for k in range(i + 1, n_frogs):
                if color_wheel_indices[k, 1] == frog_1_sec:
                    secondary_is_redundant = True
                    break
        
        # And breed with every other frog in the wheel ahead of it
        for j in range(i + 1, n_frogs):
            num_breeds = 0
            frog_2_base, frog_2_sec = color_wheel_indices[j]
            
            # Check if the 2 unique offspring are in the table
            while not (frog_table[frog_1_base, frog_2_sec] and frog_table[frog_2_base, frog_1_sec]):
                # Breed frogs, and update table 
                offspring_base, offspring_secondary = breed_pair_numba(
                    frog_1_base, frog_1_sec, frog_2_base, frog_2_sec
                )
                frog_table[offspring_base, offspring_secondary] = True
                num_breeds += 1

            # If frog_1's secondary is redundant, and we got frog_2's secondary color we can quit early
            if secondary_is_redundant and frog_table[frog_1_base, frog_2_sec]:
                continue

            base_breeds += num_breeds

        total_breeds += base_breeds

    return total_breeds

def create_frog_table():
    """Create and populate the initial frog table with color wheel frogs"""
    frog_table = np.zeros((len(base_colors), len(secondary_colors)), dtype=bool)
    
    # Populate with color wheel
    for base, secondary in color_wheel:
        base_idx = base_to_idx[base]
        secondary_idx = secondary_to_idx[secondary]
        frog_table[base_idx, secondary_idx] = True
    
    return frog_table

def validate_setup(frog_table):
    """Validate that the setup is correct"""
    # Check to make sure color wheel covers all colors
    assert np.all(frog_table.sum(axis=1) > 0), "Not all base colors covered by color wheel"
    assert np.all(frog_table.sum(axis=0) > 0), "Not all secondary colors covered by color wheel"
    print("Setup validation passed")

def run_simulation(n=500):
    """Run the complete simulation"""
    print("Creating frog table")
    base_frog_table = create_frog_table()
    
    print("Validating setup")
    validate_setup(base_frog_table)
    
    print(f"Number of simulations per trial: {n}")
    
    # Pre-allocate results arrays
    strategy_1_results = np.zeros(n)
    strategy_2_results = np.zeros(n)
    
    print("Running strategy 1")
    start_time = time.time()
    for i in range(n):
        frog_table_copy = base_frog_table.copy()
        strategy_1_results[i] = strategy_1_numba(frog_table_copy, color_wheel_indices)
        
        # Validate that we got all frogs (368 total)
        assert frog_table_copy.sum() == 368, f"Strategy 1 trial {i} does not get every single breed"
    
    strategy_1_time = time.time() - start_time
    print(f"Finished strategy 1 in {strategy_1_time:.2f} seconds")

    print("Running strategy 2")
    start_time = time.time()
    for i in range(n):
        frog_table_copy = base_frog_table.copy()
        strategy_2_results[i] = strategy_2_numba(frog_table_copy, color_wheel_indices)
        
        # Validate that we got all frogs (368 total)
        assert frog_table_copy.sum() == 368, f"Strategy 2 trial {i} does not get every single breed"
    
    strategy_2_time = time.time() - start_time
    print(f"Finished strategy 2 in {strategy_2_time:.2f} seconds")

    # Print results
    print(f"\nAverage number of breeding events for strategy 1: {np.mean(strategy_1_results):.2f}")
    print(f"Standard deviation of breeding events for strategy 1: {np.std(strategy_1_results):.2f}")
    print(f"Strategy 1 execution time: {strategy_1_time:.2f} seconds\n")

    print(f"Average number of breeding events for strategy 2: {np.mean(strategy_2_results):.2f}")
    print(f"Standard deviation of breeding events for strategy 2: {np.std(strategy_2_results):.2f}")
    print(f"Strategy 2 execution time: {strategy_2_time:.2f} seconds")

if __name__ == "__main__":
    run_simulation()
