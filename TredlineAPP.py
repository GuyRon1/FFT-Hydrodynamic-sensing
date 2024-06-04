import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from the CSV file into a pandas DataFrame
df = pd.read_csv('Q=60LPM,CYL=TRUE(2).csv')

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

# Calculate the mean value of pressure for each sensor
mean_pressure_1 = df['Pressure [Pa] - P#1'].mean()
mean_pressure_2 = df['Pressure [Pa] - P#2'].mean()
mean_pressure_3 = df['Pressure [Pa] - P#3'].mean()

# Subtract the mean value from all pressure values for each sensor
df['Pressure [Pa] - P#1'] -= mean_pressure_1
df['Pressure [Pa] - P#2'] -= mean_pressure_2
df['Pressure [Pa] - P#3'] -= mean_pressure_3

# Plotting the data in subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Plot for Sensor 1
axs[0, 0].plot(df['Flow time [sec]  - P#1'], df['Pressure [Pa] - P#1'], label='Sensor 1')
axs[0, 0].set_ylabel('Voltage (V)')
axs[0, 0].set_title('Sensor 1 - Pressure as function of time')
axs[0, 0].axhline(y=0, color='black', linestyle='--', label=f'Trendline = {mean_pressure_1:.2f}')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot for Sensor 2
axs[1, 0].plot(df['Flow time [sec]  - P#2'], df['Pressure [Pa] - P#2'], label='Sensor 2')
axs[1, 0].set_ylabel('Voltage (V)')
axs[1, 0].set_title('Sensor 2 - Pressure as function of time')
axs[1, 0].axhline(y=0, color='black', linestyle='--', label=f'Trendline = {mean_pressure_2:.2f}')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot for Sensor 3
axs[2, 0].plot(df['Flow time [sec]  - P#3'], df['Pressure [Pa] - P#3'], label='Sensor 3')
axs[2, 0].set_xlabel('Time (sec)')
axs[2, 0].set_ylabel('Voltage (V)')
axs[2, 0].set_title('Sensor 3 - Pressure as function of time')
axs[2, 0].axhline(y=0, color='black', linestyle='--', label=f'Trendline = {mean_pressure_3:.2f}')
axs[2, 0].legend()
axs[2, 0].grid(True)

# Magnified view for Sensor 1
axs[0, 1].plot(df['Flow time [sec]  - P#1'], df['Pressure [Pa] - P#1'], label='Sensor 1')
axs[0, 1].set_ylabel('Voltage (V)')
axs[0, 1].set_xlabel('Time (sec)')
axs[0, 1].set_title('Sensor 1 - Magnified View (300-400 sec)')
axs[0, 1].axhline(y=0, color='black', linestyle='--', label=f'Trendline = {mean_pressure_1:.2f}')
axs[0, 1].legend()
axs[0, 1].grid(True)
axs[0, 1].set_xlim(300, 400)
axs[0, 1].set_xticks(np.arange(300, 401, 5))

# Magnified view for Sensor 2
axs[1, 1].plot(df['Flow time [sec]  - P#2'], df['Pressure [Pa] - P#2'], label='Sensor 2')
axs[1, 1].set_ylabel('Voltage (V)')
axs[1, 1].set_xlabel('Time (sec)')
axs[1, 1].set_title('Sensor 2 - Magnified View (300-400 sec)')
axs[1, 1].axhline(y=0, color='black', linestyle='--', label=f'Trendline = {mean_pressure_2:.2f}')
axs[1, 1].legend()
axs[1, 1].grid(True)
axs[1, 1].set_xlim(300, 400)
axs[1, 1].set_xticks(np.arange(300, 401, 5))

# Magnified view for Sensor 3
axs[2, 1].plot(df['Flow time [sec]  - P#3'], df['Pressure [Pa] - P#3'], label='Sensor 3')
axs[2, 1].set_ylabel('Voltage (V)')
axs[2, 1].set_xlabel('Time (sec)')
axs[2, 1].set_title('Sensor 3 - Magnified View (300-400 sec)')
axs[2, 1].axhline(y=0, color='black', linestyle='--', label=f'Trendline = {mean_pressure_3:.2f}')
axs[2, 1].legend()
axs[2, 1].grid(True)
axs[2, 1].set_xlim(300, 400)
axs[2, 1].set_xticks(np.arange(300, 401, 5))

plt.tight_layout()
plt.show()
