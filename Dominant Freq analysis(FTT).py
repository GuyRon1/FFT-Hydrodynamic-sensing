import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

# Read data from the CSV file into a pandas DataFrame
df = pd.read_csv('80LPM_CYCTRUE.CSV')

# Convert time values from "mm:ss.ms" to seconds
def time_to_seconds(time_str):
    minutes, seconds_ms = time_str.split(':')
    seconds, milliseconds = seconds_ms.split('.')
    total_seconds = int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    return total_seconds

# Apply the time_to_seconds function to convert time values to seconds for each set of data
df['Flow time [sec]  - P#1'] = df['Flow time [sec]  - P#1'].apply(time_to_seconds)
df['Flow time [sec]  - P#2'] = df['Flow time [sec]  - P#2'].apply(time_to_seconds)
df['Flow time [sec]  - P#3'] = df['Flow time [sec]  - P#3'].apply(time_to_seconds)

# Select relevant columns for plotting for each set of data
selected_columns_1 = ['Flow time [sec]  - P#1', 'Pressure [Pa] - P#1']
selected_columns_2 = ['Flow time [sec]  - P#2', 'Pressure [Pa] - P#2']
selected_columns_3 = ['Flow time [sec]  - P#3', 'Pressure [Pa] - P#3']

df_selected_1 = df[selected_columns_1]
df_selected_2 = df[selected_columns_2]
df_selected_3 = df[selected_columns_3]

# Plotting the FFT of the data with desired frequency resolution
plt.figure(figsize=(15, 10))

# Define cutoff frequency for the high-pass filter
cutoff_frequency = 0.1  # Hz

# Calculate zero-padding factor
original_num_samples = len(df_selected_1)
desired_freq_resolution = 0.005  # Hz
sampling_interval = 0.5  # seconds (constant)
zero_padding_factor = int(1 / (desired_freq_resolution * sampling_interval)) - original_num_samples
zero_padding_factor = max(0, zero_padding_factor)  # Ensure non-negative value

# Increase data length (optional)
# You may collect more samples to increase data length

for i, df_selected in enumerate([df_selected_1, df_selected_2, df_selected_3], start=1):
    # Apply zero-padding to the data
    pressure_data = df_selected['Pressure [Pa] - P#{}'.format(i)].values
    pressure_data = np.pad(pressure_data, (0, zero_padding_factor), mode='constant')

    # Apply Kaiser window to the data
    kaiser_window = windows.kaiser(len(pressure_data), beta=14)  # Beta parameter can be adjusted for different sidelobe suppression
    pressure_data *= kaiser_window

    # Calculate FFT for the sensor's data with zero padding and Kaiser window
    fft_data = np.fft.fft(pressure_data)
    freq = np.fft.fftfreq(len(pressure_data), d=sampling_interval)

    # Apply high-pass filter
    fft_data_filtered = fft_data.copy()
    fft_data_filtered[np.abs(freq) < cutoff_frequency] = 0

    # Find dominant frequency
    dominant_freq_index = np.argmax(np.abs(fft_data_filtered))
    dominant_freq = np.abs(freq[dominant_freq_index])

    # Plot the result in a subplot
    plt.subplot(3, 1, i)
    plt.plot(freq, np.abs(fft_data_filtered), label=f'FFT Sensor {i} (High-pass filtered, Kaiser Window)\nDominant Frequency: {dominant_freq:.3f} Hz', color='green')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'FFT of Pressure Sensor {i} Data with High-pass Filter and Kaiser Window (Computed using np.fft.fft)')
    plt.legend()
    plt.xlim(0, 1 / (2 * sampling_interval))  # Nyquist frequency is 1 / (2 * sampling_interval)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()
