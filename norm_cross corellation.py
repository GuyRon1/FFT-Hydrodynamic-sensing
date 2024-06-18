import numpy as np

# Define the function for normalized cross-correlation
def norm_cross_correlation(I1, I2):
    # Converting I1 and I2 to numpy arrays for easier manipulation
    I1 = np.array(I1)
    I2 = np.array(I2)

    # Getting the array size
    height, width = I1.shape

    # Build the correlation matrix
    norm_corr_matrix = np.zeros((2 * height - 1, 2 * width - 1))

    for x in range(-height + 1, height):
        for y in range(-width + 1, width):
            num = 0  # numerator needed for the normalization of the cross correlation
            denom1 = 0  # denominator for image I1 (needed for normalization)
            denom2 = 0  # denominator for image I2 (needed for normalization)

            # Iterating on each pixel in image I1
            for i in range(height):
                for j in range(width):
                    if 0 <= i + x < height and 0 <= j + y < width:
                        num += I1[i, j] * I2[i + x, j + y]  # cross correlation sum without the normalization
                        # calculating needed denominators for the normalization
                        denom1 += I1[i, j] ** 2
                        denom2 += I2[i + x, j + y] ** 2

            # computing the normalized cross correlation value for the current shift
            if denom1 > 0 and denom2 > 0:  # making sure we don't divide by zero
                norm_corr_matrix[x + height - 1, y + width - 1] = num / np.sqrt(denom1 * denom2)
            else:
                norm_corr_matrix[x + height - 1, y + width - 1] = 0  # Handle division by zero case

    # Find the maximum correlation index in the correlation matrix
    max_idx = np.unravel_index(np.argmax(norm_corr_matrix), norm_corr_matrix.shape)

    # Calculate the displacement vector (dx, dy)
    displacement = (max_idx[0] - height + 1, max_idx[1] - width + 1)

    return norm_corr_matrix, displacement

# Define the input matrices I1 and I2
I1 = [
  [0, 1, 2, 1, 0],
  [1, 2, 3, 2, 1],
  [2, 3, 4, 3, 2],
  [1, 2, 3, 2, 1],
  [0, 1, 2, 1, 0]
]

I2 = [
  [0, 0, 1, 2, 1],
  [0, 1, 2, 3, 2],
  [1, 2, 3, 4, 3],
  [2, 3, 2, 1, 0],
  [1, 2, 1, 0, 0]
]

# Call the function to compute normalized cross-correlation and displacement
norm_corr_matrix, displacement = norm_cross_correlation(I1, I2)

# Print results
print("Normalized Cross-Correlation Matrix:")
print(norm_corr_matrix)
print("Displacement Vector (dx, dy):", displacement)
