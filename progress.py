"""PIV Algorithm
1) Process the images and gray scale them.
2) Loop through the images with interrogation windows.
3) For every window, compute the Fourier transforms.
4) Find the relative displacement in each position.
5) Compute the displacement vector by subtracting the peak position from the center position.
6) Return the displacement vector and plot the velocity field.
"""

import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from PIL import Image


def preprocess_image(image):
    """Convert image to numpy array and normalize."""
    return np.array(image) / 255.0


def cross_correlation_fft(window1, window2, window_size=32):
    """Compute cross-correlation using FFT."""
    f1 = fft.fft2(window1)  # FFT for window1
    f2 = fft.fft2(window2)  # FFT for window2
    f2_conj = np.conj(f2)  # Finding the conjugate of window2
    cross_corr = fft.ifft2(f1 * f2_conj)  # Multiplication in complex is convolution in spatial (correlation)
    cross_corr_abs = np.abs(cross_corr)  # Calculating the absolute value of the correlation
    width, height = cross_corr_abs.shape  # Initializing dimensions
    peak_x, peak_y = np.unravel_index(np.argmax(cross_corr_abs),
                                      cross_corr_abs.shape)  # Get indices where correlation function is maximum.
    dx = peak_x - (width // 2)  # Calculating displacement relative to the center
    dy = peak_y - (height // 2)  # Calculating displacement relative to the center
    """Note:
    Positive dx means a shift to the right.
    Negative dx means a shift to the left.
    Positive dy means a shift downward.
    Negative dy means a shift upward.
    """
    return dx, dy


def compute_cross_function(image1, image2, window_size=32, overlap=0.5):
    """Compute velocity vectors from two images."""
    step = int(window_size * (1 - overlap))  # Calculate step size based on overlap
    height, width = image1.shape  # Initialize array sizes
    """Initialize the displacement vectors"""
    x_displacement = np.zeros((height, width))
    y_displacement = np.zeros((height, width))

    """Iteration of both window 1 and 2 together with the assumption most of the particles didn’t exit the window"""
    for i in range(0, height - window_size, step):
        for j in range(0, width - window_size, step):
            window1 = image1[i:i + window_size, j:j + window_size]  # Current image1's window
            window2 = image2[i:i + window_size, j:j + window_size]  # Current image2's window
            dx, dy = cross_correlation_fft(window1, window2, window_size)  # Getting displacements
            center_i = i + window_size // 2
            center_j = j + window_size // 2
            if 0 <= center_i < width and 0 <= center_j < height:
                # Store the displacement values in the array
                x_displacement[center_i, center_j] = dx
                y_displacement[center_i, center_j] = dy
    return x_displacement, y_displacement


def plot_velocities(image1, image2, window_size=32):
    x_displacement, y_displacement = compute_cross_function(image1, image2, window_size)  # Getting displacements array
    step = window_size // 2  # Vector will be plotted every step
    height, width = image1.shape  # Get the dimensions of the image

    """Only on X,Y values will the vectors be plotted"""
    x_positions = np.arange(0, width, step)
    y_positions = np.arange(0, height, step)
    X, Y = np.meshgrid(x_positions, y_positions)

    """Populate U and V arrays with the displacements at each step"""
    U = x_displacement[::step, ::step]  # Select every step-th element for U
    V = y_displacement[::step, ::step]  # Select every step-th element for V

    # Create the quiver plot
    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='blue')
    plt.title('Velocity Field Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Set the limits of the plot to match the figure size
    plt.xlim(0, width)  # X-axis from 0 to width
    plt.ylim(height, 0)  # Y-axis from height to 0 (invert y-axis for image coordinate system)

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

# Convert images to numpy arrays
image1 = preprocess_image(image1)
image2 = preprocess_image(image2)

# Compute and plot velocity vectors
plot_velocities(image1, image2, window_size=32)  
