import serial
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# === Settings ===
SERIAL_PORT = "COM13"
BAUDRATE = 1000000
READ_BYTES = 5
COLLECT_SECONDS = 3
SAVE_BASE_DIR = "./data_softer_material"  # Base directory

# === Ensure base directory exists ===
os.makedirs(SAVE_BASE_DIR, exist_ok=True)

# === Menu Options ===
object_types = "soft_material"
height_distance_options = [
    ("height10cm", "distance10cm"),
    ("height10cm", "distance30cm"),
    ("height30cm", "distance10cm"),
    ("height30cm", "distance30cm")
]

# === Default selections ===
current_object = 'soft_material'
current_height = 'height10cm'
current_distance = 'distance10cm'


def select_height_distance():
    global current_height, current_distance
    print("Select Height/Distance Combination:")
    for idx, (h, d) in enumerate(height_distance_options):
        print(f"{idx}: {h} and {d}")
    combo_choice = int(input("Enter choice: "))
    current_height, current_distance = height_distance_options[combo_choice]


# === Cleaning Function ===
def clean_data_file(filename):
    print(f"Start Cleaning data file: {filename}")
    tmp_path = filename + '.tmp'
    lower_bound = 0
    upper_bound = 64000
    count = 0
    valid_data = []

    with open(filename, 'r') as fin, open(tmp_path, 'w') as fout:
        for row in fin:
            line = row.strip()
            # print(f"Processing line {count}: {line}")
            if line.isdigit():
                if 800 < int(line) < 4095 and (lower_bound < count < upper_bound):
                    valid_data.append(int(line))
                    fout.write(line + '\n')
            count += 1
    with open(tmp_path, 'w') as fout:
        for value in valid_data:
            # print(value)
            fout.write(f"{value}\n")

    os.replace(tmp_path, filename)
    print(f"Cleaning complete. Only valid positive integers remain in {filename}.")

def plot_fft(filename):
    # Step 1: Load the data
    dataset = np.loadtxt(filename)  # Adjust path if needed
    
    # Step 2: Define the sample rate and time axis
    sample_rate = 10000  # in Hz
    num_samples = len(dataset)

    # Step 3: Perform FFT
    fft_result = np.fft.fft(dataset)
    fft_freqs = np.fft.fftfreq(num_samples, d=1/sample_rate)

    # Step 4: Calculate the magnitude of the FFT
    fft_magnitude = np.abs(fft_result)

    # Step 5: Only keep the positive frequencies
    positive_freqs = fft_freqs[:num_samples // 2]
    positive_magnitude = fft_magnitude[:num_samples // 2]

    # Step 6a: Find the top 10 dominant frequencies
    top_freq_indices = np.argsort(positive_magnitude)[-51:-1][::-1]  # Sort descending
    top_frequencies = positive_freqs[top_freq_indices]
    top_magnitudes = positive_magnitude[top_freq_indices]

    # Step 7: Plot (optional visualization)
    plt.figure(figsize=(10, 6))
    plt.plot(positive_freqs[1:], positive_magnitude[1:])
    plt.title('FFT of ADC Data')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    # Mark top 10 dominant frequencies
    for i in range(len(top_frequencies)):
        plt.plot(top_frequencies[i], top_magnitudes[i], 'ro')  # Red dots
        # print(f"Top {i+1} Frequency: {top_frequencies[i]:.2f} Hz, Magnitude: {top_magnitudes[i]:.2f}")
    plt.show()
    
def plot_time(filename):
    # Read and convert to integers
    with open(filename, 'r') as f:
        adc_values = [int(line.strip()) for line in f if line.strip().isdigit()]
    print(f"Total data points: {len(adc_values)}")
    # print(f"ADC values: {adc_values}")
    # Generate x values (data index)
    x_values = list(range(len(adc_values)))
    time_stamp = 1/10000  # Sample rate in Hz
    # x_values = [i * time_stamp for i in x_values]  # Convert to time in seconds
    voltage_value = [v * 3.3 / 4095 for v in adc_values]  # Convert ADC values to voltage

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, adc_values, marker='o', linestyle='-', markersize=2)
    # plt.plot(x_values, voltage_value, marker='x', linestyle='-', markersize=2, color='red')
    plt.title("ADC Data Plot")
    plt.xlabel("Sample Number")
    plt.ylabel("ADC Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    
# === Main collection function ===
def collect_data(object_type, height, distance):
    dir_path = os.path.join(SAVE_BASE_DIR, object_type, height, distance)
    os.makedirs(dir_path, exist_ok=True)

    existing_files = [f for f in os.listdir(dir_path) if f.startswith("adc_") and f.endswith(".data")]
    file_number = len(existing_files) + 1

    filename = os.path.join(dir_path, f"adc_{object_type}_{height}_{distance}_{file_number}.data")

    print(f"Saving ADC data to {filename}")

    ser = serial.Serial(port=SERIAL_PORT, baudrate=BAUDRATE, bytesize=8, parity="N", stopbits=1, timeout=0.01)

    count = 0

    with open(filename, "wb") as file_1:
        try:
            start_time = time.time()
            while (time.time() - start_time) < COLLECT_SECONDS:
                data = ser.read(READ_BYTES)
                if data:
                    file_1.write(data)
                    print(f"{count}: Received {data}")
                    count += 1

        except KeyboardInterrupt:
            print("\nUser stopped the data collection.")

        finally:
            ser.close()
            elapsed_time = time.time() - start_time
            print(f"Data collection time: {elapsed_time:.5f} seconds")
            print("Serial connection closed.")
            print(f"Total data points collected: {count}")

    # Cleaning step after data collection
    try:
        clean_data_file(filename)
        plot_time(filename)
        plot_fft(filename)
    except Exception as e:
        print(f"Cleaning failed: {e}")

# === Program Entry ===
if __name__ == "__main__":
    while True:
        print("\n=== MAIN MENU ===")
        print(f"Current Object: {current_object}")
        print(f"Current Height: {current_height}")
        print(f"Current Distance: {current_distance}")
        print("1. Start Collecting Data")
        print("2. Change Height/Distance")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            collect_data(current_object, current_height, current_distance)
        elif choice == '2':
            select_height_distance()
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")