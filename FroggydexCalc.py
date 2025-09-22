import pandas as pd
import numpy as np
import random

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

def breed_pair(frog_1: tuple, frog_2: tuple):
    # Function to create random offspring from 2 frogs
    base_colors = [frog_1[0], frog_2[0]]
    secondary_colors = [frog_1[1], frog_2[1]]
    return (base_colors[random.randint(0, 1)], secondary_colors[random.randint(0, 1)])

# In this strategy we will go through the color wheel and breed each frog with every other frog
# We will continue breeding until we get the 2 unique offspring for that pair
# If the offspring are redundant (i.e. breeding Maroon Tingo with Red Tingo) we will just move on
# If we already have the offspring from a previous breeding, we move on
def strategy_1(frog_table: pd.DataFrame, color_wheel: list):
    total_breeds = 0

    # Loop across every frog in the color wheel
    for frog_1 in color_wheel:
        base_breeds = 0
        # And breed with every other frog in the wheel ahead of it
        for frog_2 in color_wheel[color_wheel.index(frog_1) + 1:]:
            num_breeds = 0  # breeding events for this specific pair
            base_1, secondary_1 = frog_1
            base_2, secondary_2 = frog_2
            # Check if the 2 unique offspring are in the table
            while not (frog_table.loc[base_1, secondary_2] and frog_table.loc[base_2, secondary_1]):
                # Breed frogs, and update table 
                offspring_base, offspring_secondary = breed_pair(frog_1, frog_2)
                frog_table.loc[offspring_base, offspring_secondary] = True
                num_breeds += 1

            base_breeds += num_breeds  # breeding events for the whole base color (i.e. Maroon)
            # print(f"Breeds to complete {frog_1[0]}: {base_breeds}")  # line for testing

        total_breeds += base_breeds

    assert frog_table.values.sum() == 368, "Strategy does not get every single breed"

    return total_breeds


# Strategy is very similar to strategy_1 with one exception
# For a frog with a redundant base colors in our color wheel, i.e. Maroon Tingo and Red Tingo,
# if we get the first frog's base color with the other frog's secondary color, we move on 
# since the first frog's secondary color with the other frog's base will be achieved with the redundant color wheel frog
# As an example:
# When breeding Maroon Tingo with Purple Pruni, we want Maroon Pruni and Purple Tingo
# if, by luck/chance, we get Maroon Pruni before Purple Tingo, we just move on
# this is because we will get Purple Tingo when we breed Red Tingo with Purple Pruni
# so in theory we are using the good luck of the first breeding
def strategy_2(frog_table: pd.DataFrame, color_wheel: list):
    total_breeds = 0

    # Loop across every frog in the color wheel
    for frog_1 in color_wheel:
        base_breeds = 0

        # Check if the secondary color is redundant 
        future_frogs = np.array(color_wheel[color_wheel.index(frog_1) + 1:])
        secondary_is_redundant = len(future_frogs) > 1 and frog_1[1] in future_frogs[:, 1]
        # if secondary_is_redundant: print(f"{frog_1[1]} is redundant in {frog_1}")  # line for testing

        # And breed with every other frog in the wheel ahead of it
        for frog_2 in color_wheel[color_wheel.index(frog_1) + 1:]:
            num_breeds = 0  # breeding events for this specific pair
            base_1, secondary_1 = frog_1
            base_2, secondary_2 = frog_2
            # Check if the 2 unique offspring are in the table
            while not (frog_table.loc[base_1, secondary_2] and frog_table.loc[base_2, secondary_1]):
                # Breed frogs, and update table 
                offspring_base, offspring_secondary = breed_pair(frog_1, frog_2)
                frog_table.loc[offspring_base, offspring_secondary] = True
                num_breeds += 1

            # If frog_1's secondary is redundant, and we got frog_2's secondary color we can quit early
            if secondary_is_redundant and frog_table.loc[base_1, secondary_2]:
                continue

            base_breeds += num_breeds  # breeding events for the whole base color (i.e. Maroon)
            # print(f"Breeds to complete {frog_1[0]}: {base_breeds}")  # line for testing

        total_breeds += base_breeds

    assert frog_table.values.sum() == 368, "Strategy does not get every single breed"

    return total_breeds


# --------------#
if __name__ == "__main__":
    # Create empty table, True represents we have the frog, False if we don't
    # Populate with the color wheel
    print("Creating frog table")
    frog_table = pd.DataFrame(False, index=base_colors, columns=secondary_colors)

    print("Populating with color wheel")
    for frog in color_wheel:
        base, secondary = frog
        if base not in base_colors or secondary not in secondary_colors:
            raise Exception("Color wheel contains invalid frog color")
        frog_table.loc[base, secondary] = True

    # Check to make sure color wheel covers all colors
    print("Checking table")
    assert 0 not in frog_table.sum(axis=1).values, "Not all base colors covered by color wheel"
    assert 0 not in frog_table.sum(axis=0).values, "Not all secondary colors covered by color wheel"

    # Run n trials for the strategy to determine average number of events needed
    n = 500
    print(f"Number of simulations per trial: {n}")

    print("Running strategy 1")
    strategy_1_results = np.zeros(n)
    for i in range(0, n):
        strategy_1_results[i] = strategy_1(frog_table.copy(), color_wheel)
    print("Finished strategy 1")

    print("Running strategy 2")
    strategy_2_results = np.zeros(n)
    for i in range(0, n):
        strategy_2_results[i] = strategy_2(frog_table.copy(), color_wheel)
    print("Finished strategy 2")

    print(f"Average number of breeding events for strategy 1: {np.mean(strategy_1_results):.2f}")
    print(f"Standard deviation of breeding events for strategy 1: {np.std(strategy_1_results):.2f}\n")

    print(f"Average number of breeding events for strategy 2: {np.mean(strategy_2_results):.2f}")
    print(f"Standard deviation of breeding events for strategy 2: {np.std(strategy_2_results):.2f}\n")

