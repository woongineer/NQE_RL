import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def save_probability_animation(prob_list, filename="animation.mp4"):
    """
    Creates an animation from a probability distribution dictionary and saves it as a video file.

    Parameters:
        prob_list (dict): A dictionary where keys represent frames and values contain 'prob' (list of probabilities).
        filename (str): Name of the output video file (default is 'animation.mp4').
    """
    # Extract frames and distributions
    frames = list(prob_list.keys())
    distributions = [prob_list[frame]['prob'] for frame in frames]

    # Initialize figure and bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_container = ax.bar(range(len(distributions[0])), distributions[0], color='skyblue', edgecolor='black')

    # Set axis properties
    ax.set_ylim(0, max(max(d) for d in distributions) * 1.2)
    ax.set_xlabel("Layer Index", fontsize=14)
    ax.set_ylabel("Probability", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Update function for animation
    def update(frame_idx):
        for bar, height in zip(bar_container, distributions[frame_idx]):
            bar.set_height(height)
        ax.set_title(f"Probability Distribution - Epoch {frames[frame_idx]}", fontsize=16)

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(frames), interval=650, repeat=True)

    # Save animation as a video file
    ani.save(filename, writer="ffmpeg")
    plt.close(fig)
