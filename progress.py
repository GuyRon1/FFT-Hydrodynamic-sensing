import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import griddata

def preprocess_image(image):
    """Convert image to numpy array and normalize."""
    return np.array(image) / 255.0

def cross_correlation_fft(window1, window2):
    """Compute cross-correlation using FFT."""
    f1 = fft.fft2(window1)
    f2 = fft.fft2(window2)
    f2_conj = np.conj(f2)
    cross_corr = fft.ifft2(f1 * f2_conj)
    cross_corr = np.abs(cross_corr).max().item()
    return cross_corr

"""Explanation about interrogation windows and their role in PIV."""
# In Particle Image Velocimetry (PIV), images are divided into smaller subregions called interrogation windows.
# These windows are small, square regions of the original image used to determine local particle displacement between successive images.

"""Explanation on the appearance of interrogation windows."""
# An interrogation window is a small section of the larger image, containing a subset of pixels.

"""Explanation on how interrogation windows are chosen."""
# Interrogation windows are selected by dividing the entire image into a grid of equally sized smaller regions.
# Parameters such as size and overlap of these windows are crucial in PIV analysis.
# Size: Chosen based on the scale of flow features of interest.
# Overlap: Often used to provide smoother and more continuous velocity fields, e.g., 50% overlap means windows overlap by half their size.

"""Explanation of why displacement is returned."""
# The cross-correlation matrix (cross_corr) indicates similarity values for each relative position (shift) of two windows.
# The position with the highest correlation corresponds to the displacement of particles between images.
# The center of the cross-correlation matrix shows no displacement (zero shift).
# Displacement vector is calculated by measuring peak offset from this center, indicating how much the second window shifts to align with the first.

def compute_cross_function(image1, image2, window_size=16, overlap=0.5):
    """Compute velocity vectors from two images."""
    step = int(window_size * (1 - overlap))
    velocities = []

    # Convert images to numpy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)
    max_index_dictionary = {}
    # Iterate over possible starting positions for window1 in image1
    for i in range(0, image1.shape[0] - window_size, step):
        for j in range(0, image1.shape[1] - window_size, step):
            # Fix window1 at position (i, j) in image1
            window1 = image1[i:i + window_size, j:j + window_size]

            # Initialize list to store cross-correlations for window1
            cross_corrs = {}

            # Iterate over possible starting positions for window2 in image2
            for k in range(0, image2.shape[0] - window_size, step):
                for l in range(0, image2.shape[1] - window_size, step):
                    # Fix window2 at position (k, l) in image2
                    window2 = image2[k:k + window_size, l:l + window_size]

                    # Compute cross-correlation between window1 and window2
                    cross_corr = cross_correlation_fft(window1, window2)

                    # Store cross-correlation result
                    cross_corrs[(k,l)]=cross_corr
            max_key, max_value = max(cross_corrs.items(), key=lambda item: item[1])
            max_index_dictionary[(i,j)]=max_key
    print(max_index_dictionary)
    return max_index_dictionary

def compute_velocity(image1, image2):
    max_index_dictionary = compute_cross_function(image1, image2)

    # Extract original and displaced positions from the dictionary
    original_placements = np.array(list(max_index_dictionary.keys()))
    displacements = np.array(list(max_index_dictionary.values())) - original_placements

    # Combine original placements and displacements
    displacements_per_position = np.column_stack((original_placements, displacements))
    print("Just found displacement")
    # Extract x and y positions and displacements from the combined array (assuming first two elements are relevant)
    # Combine original placements and displacements
    displacements_per_position = np.column_stack((original_placements, displacements))
    print("Just found displacement")
    # Extract x and y positions and displacements from the combined array (assuming first two elements are relevant)
    positions = displacements_per_position[:, :2]  # Select first two columns
    displacements = displacements_per_position[:, 2:]  # Select remaining columns
    x_positions = positions[:,0]
    y_positions = positions[:,1]
    dx = displacements[:,0]
    dy =displacements[:,1]
    print("ready to plot")

    # Create the plot
    plt.figure(figsize=(10, 10))
    plt.quiver(x_positions, y_positions, dx, dy, angles='xy', scale_units='xy', scale=1, color='b')
    plt.xlim(0, 256)
    plt.ylim(0, 256)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Vector Field')
    plt.gca().invert_yaxis()  # Invert y-axis to match the image coordinate system
    plt.grid(True)
    plt.show()
    print("done plot vector")

    # Convert positions and displacements to numpy arrays
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)
    dx_displacement = np.array(dx)
    dy_displacement = np.array(dy)

    # Create a grid based on the original x and y positions
    x_unique = np.sort(np.unique(x_positions))
    y_unique = np.sort(np.unique(y_positions))
    x_grid, y_grid = np.meshgrid(x_unique, y_unique)

    # Initialize displacement grids
    u_grid = np.zeros_like(x_grid, dtype=float)
    v_grid = np.zeros_like(y_grid, dtype=float)

    #   Populate the displacement grids by mapping positions to grid indices
    for i in range(len(x_positions)):
        x_idx = np.where(x_unique == x_positions[i])[0][0]
        y_idx = np.where(y_unique == y_positions[i])[0][0]
        u_grid[y_idx, x_idx] = dx_displacement[i]
        v_grid[y_idx, x_idx] = dy_displacement[i]

    # Plot the streamline
    plt.figure(figsize=(10, 10))
    plt.streamplot(x_grid, y_grid, u_grid, v_grid, density=1.5)
    plt.title('Streamline Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()  # Invert y-axis to match the image coordinate system
    plt.grid(True)
    plt.show()
    print("plotting streamline")
####################################################################

# Example usage with two images
image_path = ['piv01_1.bmp', 'piv01_2.bmp']
image1 = Image.open(image_path[0]).convert('L')
image2 = Image.open(image_path[1]).convert('L')

# Compute velocity vectors
velocities = compute_velocity(image1, image2)