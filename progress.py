import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import griddata

def preprocess_image(image):
    """Convert image to numpy array and normalize."""
    return np.array(image) / 255.0

def cross_correlation_fft(window1, window2,window_size=32):
    """Compute cross-correlation using FFT."""
    f1 = fft.fft2(window1)
    f2 = fft.fft2(window2)
    f2_conj = np.conj(f2)
    cross_corr = fft.ifft2(f1 * f2_conj)
    cross_corr_abs = np.abs(cross_corr) #return the index with the max abs value
    width,height = cross_corr_abs.shape
    peak_x,peak_y = np.unravel_index(np.argmax(cross_corr_abs), cross_corr_abs.shape) #return a tuple with the indices of the maximum value in the cross_matrix

    dx = peak_x - (width // 2)
    dy = peak_y - (height // 2)
    return dx ,dy

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

def compute_cross_function(image1, image2, window_size=32, overlap=0.5):
    """Compute velocity vectors from two images."""
    #step = int(window_size * (1 - overlap))
    step=window_size
    height, width = image1.size
    # Convert images to numpy arrays
    image1 = np.array(image1)
    image2 = np.array(image2)
    displacement = np.zeros((image1.shape),dtype=object) #Initialize a velocity array
    x_displacement = np.zeros((height, width))
    y_displacement = np.zeros((height, width))
    """Iteration of both window 1 and 2 together with the assumption most of the particals didnt exit the window"""
    for i in range(0, image1.shape[0] - window_size, step):
        for j in range( 0, image1.shape[1] - window_size, step):
            # Fix window1 at position (i, j) in image1
            window1 = image1[i:i + window_size, j:j + window_size]
            window2 = image2[i:i + window_size, j:j + window_size]
            dx, dy=cross_correlation_fft(window1, window2) #indecies with the max displacment
            #x_velocities[i]=i+shift_x #positive velocity is right
            #y_velocities=j+shift_y #positive velocity is down
            # Compute the center position of the current window in the 256x256 grid
            center_i = i + window_size // 2
            center_j = j + window_size // 2
            if 0 <= center_i < width and 0 <= center_j < height:
                # Store the displacement values
                x_displacement[center_i, center_j] = dx
                y_displacement[center_i, center_j] = dy
            #y_displacement[i + window_size // 2, j + window_size // 2] = shift_y - window_size // 2
            #displacement[(i+window_size//2,j+window_size//2)]=(i+shift_x, j+shift_y) #finding displecment in the middle of window position
    return x_displacement,y_displacement

def plot_velocities(image1 , image2, window_size=32):
    x_displacement,y_displacement = compute_cross_function(image1, image2)
    step=window_size//2
    height, width = image1.size
    # Create a 256x256 figure
    plt.figure(figsize=(8, 8))
    plt.title('Velocity Field Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Create a grid of positions where the vectors will be plotted
    x_positions = np.arange(0, width, step)
    y_positions = np.arange(0, height, step)
    X, Y = np.meshgrid(x_positions, y_positions)

    # Extract the vectors to be plotted at every 16x16 grid
    U = x_displacement[::step, ::step]  # Select every step-th element
    V = y_displacement[::step, ::step]  # Select every step-th element

    # Create the quiver plot
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='blue')

    # Set the limits of the plot to match the figure size
    plt.xlim(0, height)
    plt.ylim(0, width)

    # Add a grid and adjust the aspect ratio
    plt.grid()
    plt.gca().set_aspect('equal', adjustable='box')

    # Show the plot
    plt.show()

####################################################################

# Example usage with two images
image_path = ['F_00001.bmp', 'F_00002.bmp']
image1 = Image.open(image_path[0]).convert('L')
image2 = Image.open(image_path[1]).convert('L')

# Compute velocity vectors
velocities = plot_velocities(image1, image2)
