import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

# Hamming distance
def hamming_distance(x, ref):
    return np.sum(np.array(list(x)) != np.array(list(ref)))

# Transition count (for 1D string from 2D grid)
def transition_count(x):
    return np.sum([1 for i in range(len(x)-1) if x[i] != x[i+1]])

# Combined complexity measure
def complexity_ht(x, alpha=0.5):
    n = len(x)
    s0 = "0" * n
    s1 = "1" * n
    h0 = hamming_distance(x, s0) / n
    h1 = hamming_distance(x, s1) / n
    t = transition_count(x) / (n - 1)
    c0 = alpha * h0 + (1 - alpha) * t
    c1 = alpha * h1 + (1 - alpha) * t
    return min(c0, c1)

# Conway's Game of Life update
def game_of_life(grid):
    n, m = grid.shape
    new_grid = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            # Count live neighbors
            neighbors = 0
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = (i + di) % n, (j + dj) % m
                    neighbors += grid[ni, nj]
            # Apply rules
            if grid[i, j] == 1:  # Live cell
                new_grid[i, j] = 1 if neighbors in [2, 3] else 0
            else:  # Dead cell
                new_grid[i, j] = 1 if neighbors == 3 else 0
    return new_grid

# Evolve Game of Life
def evolve_game_of_life(initial_grid, steps):
    n, m = initial_grid.shape
    states = [initial_grid.copy()]  # Store NumPy arrays for visualization
    current = initial_grid.copy()
    for _ in range(steps):
        current = game_of_life(current)
        states.append(current.copy())  # Store each grid state
    return states

# Function to update the animation frame
def update(frame, states, ax, im):
    ax.clear()
    sns.heatmap(states[frame], ax=ax, cmap='binary', cbar=False, xticklabels=False, yticklabels=False)
    ax.set_title(f"Conway's Game of Life - Step {frame}")
    return im,

# Simulate original and perturbed evolution with visualization
try:
    n, m = 30, 30  # Smaller grid for faster animation (adjust as needed)
    steps = 100  # Fewer steps for testing animation (adjust to 1,000 later)
    # Random initial state (50% alive)
    initial = np.random.choice([0, 1], size=(n, m), p=[0.5, 0.5])

    # Original evolution
    original_states = evolve_game_of_life(initial, steps)
    original_complexities = [complexity_ht(''.join(state.flatten().astype(str))) for state in original_states]

    # Perturbed evolution (flip 5 random cells after initial evolution)
    perturbed_grid = original_states[-1].copy()
    # Flip 5 random cells
    for _ in range(5):
        i, j = np.random.randint(0, n), np.random.randint(0, m)
        perturbed_grid[i, j] = 1 - perturbed_grid[i, j]  # Flip 0 to 1 or 1 to 0
    perturbed_states = evolve_game_of_life(perturbed_grid, steps)
    perturbed_complexities = [complexity_ht(''.join(state.flatten().astype(str))) for state in perturbed_states]

    # Plot original complexity evolution
    plt.figure(figsize=(12, 6))
    plt.plot(original_complexities, label="Original C_HT", color='blue')
    plt.xlabel("Time Step")
    plt.ylabel("Complexity (C_HT)")
    plt.title("Complexity Evolution in Conway's Game of Life (Original, 100 Steps)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot comparison of original and perturbed complexities
    plt.figure(figsize=(12, 6))
    plt.plot(original_complexities, label="Original C_HT", color='blue')
    plt.plot(perturbed_complexities, label="Perturbed C_HT", color='red', linestyle='--')
    plt.xlabel("Time Step")
    plt.ylabel("Complexity (C_HT)")
    plt.title("Complexity Evolution: Original vs. Perturbed (100 Steps)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Visualize grid evolution with animation (original)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(original_states[0], cmap='binary')
    ani = FuncAnimation(fig, update, frames=len(original_states), fargs=(original_states, ax, im), interval=200, blit=False)
    plt.title("Conway's Game of Life Evolution (Original, 100 Steps)")
    plt.close()  # Close the figure to prevent immediate display
    ani.save('gol_evolution_original.gif', writer='pillow')  # Save as GIF (requires pillow)
    print("Animation saved as 'gol_evolution_original.gif'")

    # Visualize grid evolution with animation (perturbed)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(perturbed_states[0], cmap='binary')
    ani = FuncAnimation(fig, update, frames=len(perturbed_states), fargs=(perturbed_states, ax, im), interval=200, blit=False)
    plt.title("Conway's Game of Life Evolution (Perturbed, 100 Steps)")
    plt.close()  # Close the figure to prevent immediate display
    ani.save('gol_evolution_perturbed.gif', writer='pillow')  # Save as GIF
    print("Animation saved as 'gol_evolution_perturbed.gif'")

    # Visualize final states (heatmaps)
    final_original = original_states[-1]
    plt.figure(figsize=(8, 8))
    sns.heatmap(final_original, cmap='binary', cbar=False)
    plt.title("Final State (Original, Step 100)")
    plt.show()

    final_perturbed = perturbed_states[-1]
    plt.figure(figsize=(8, 8))
    sns.heatmap(final_perturbed, cmap='binary', cbar=False)
    plt.title("Final State (Perturbed, Step 100)")
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")
