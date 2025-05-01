import numpy as np
import matplotlib.pyplot as plt

# ========== User Input ==========
label = input("Enter label for this dataset: ")  # Let user type label
output_file = './dataset_collection.txt'         # File to save features
# ================================

# Step 1: Load the data
dataset = np.loadtxt('./adc.data')  # Adjust path if needed
    
# Step 2: Define the sample rate and time axis
sample_rate = 12800  # in Hz
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
top_freq_indices = np.argsort(positive_magnitude)[-101:-1][::-1]  # Sort descending
top_frequencies = positive_freqs[top_freq_indices]
top_magnitudes = positive_magnitude[top_freq_indices]

# Step 6b: Find the top 10 highest amplitude values (time domain)
top_amp_indices = dataset.argsort()[-20:][::-1]  # Sort descending
top_amplitudes = dataset[top_amp_indices]

low_amp_indices = dataset.argsort()[:20]  # Sort ascending  
low_amplitudes = dataset[low_amp_indices]


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
    print(f"Top {i+1} Frequency: {top_frequencies[i]:.2f} Hz, Magnitude: {top_magnitudes[i]:.2f}")
plt.show()

for i in range(len(top_amplitudes)):
    print(f"Top {i+1} Amplitude: {top_amplitudes[i]:.2f}")
    print(f"Low {i+1} Amplitude: {low_amplitudes[i]:.2f}")
    
# Step 8: Prepare the feature row (top 10 amplitudes + top 10 frequencies + label)
features = list(top_amplitudes) + list(top_frequencies) + list(top_magnitudes) + list(low_amplitudes)
features.append(label)  # Add the label at the end

# Step 9: Save (append) the features into the output file
with open(output_file, 'a') as fout:
    feature_row = ','.join(map(str, features))  # CSV style
    fout.write(feature_row + '\n')

print(f"Feature row appended successfully into {output_file}.")
