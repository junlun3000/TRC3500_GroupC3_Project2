import serial
import os
import time

# Configure serial port
ser = serial.Serial(port="COM13", baudrate=1000000, bytesize=8, parity="N", stopbits=1, timeout=0.01)

# Generate a unique filename with a timestamp to prevent overwriting
filename = f"./adc.data"

# Check if file exists, and clear it if it does
if os.path.exists(filename):
    with open(filename, "w"):  # Open in write mode to clear contents
        pass  # File is now empty

print(f"Saving ADC data to {filename}")
count = 0
# Open the file in append-binary mode
with open(filename, "ab") as file_1:
    try:
        start_time = time.time()  # Start time for data collection
        
        while True:
            data = ser.read(5)  # Read up to 5 bytes at a time

            if data:  # If data is received, write it to file
                file_1.write((data))
                print(f"{count}: Received {data}")
                count += 1
            
    except KeyboardInterrupt:
        print("\nUser stopped the data collection.")

    finally:
        ser.close()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Data collection time: {elapsed_time:.5f} seconds")
        print("Serial connection closed.")
        print(f"Total data points collected: {count}")
        

input_path  = './adc.data'
tmp_path    = input_path + '.tmp'

lower_bound = 0
upper_bound = 64000
count = 0

valid_data = []

with open(input_path, 'r') as fin, open(tmp_path, 'w') as fout:
    for raw in fin:
        line = raw.strip()

        # Check if the line is a valid positive integer
        if line.isdigit():
            if 500 < int(line) < 3850 and (lower_bound < count < upper_bound):
                # Write the valid line to the temporary file
                valid_data.append(int(line))
                fout.write(line + '\n')
        count += 1

    
# Step 3: Save filtered result back into file
with open(tmp_path, 'w') as fout:
    for value in valid_data:
        fout.write(f"{value}\n")
        
# Replace original file with the cleaned one
os.replace(tmp_path, input_path)

print("Cleaning complete. Only valid positive integers remain in adc.data.")