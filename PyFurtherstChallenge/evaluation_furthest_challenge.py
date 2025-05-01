from fft_extract_furthest_challenge import extract_features_from_adc, get_features_row
import serial
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# === Settings ===
SERIAL_PORT = "COM13"
BAUDRATE = 1000000
READ_BYTES = 5
COLLECT_SECONDS = 3

FILENAME = "evaluation_furthest_challenge.data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepMLP(nn.Module):
    def __init__(self, input_size=36, hidden_sizes=[128, 64, 32], output_size=8, p_dropout=0.3, device='cpu'):
        super().__init__()
        self.device = device

        layers = []
        in_feats = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_feats, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(p_dropout)
            ]
            in_feats = h
        layers.append(nn.Linear(in_feats, output_size))
        self.net = nn.Sequential(*layers)
        self.to(device=device)
        self.lossfn = nn.CrossEntropyLoss()
        

    def forward(self, x):
        return self.net(x)
    
    def Train(self, epochs, optimizer, loader_train, loader_test, verbose=True, patience=10):
        best_loss = float('inf')  # Initialize best loss
        self.loss_train_log = []
        self.loss_test_log = []
        epoch_no_improvement = 0  # Counter for early stopping
        for epoch in range(epochs):
            self.train()
            for x, y in loader_train:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.forward(x)
                loss = self.lossfn(y_pred, y)
                loss.backward()
                optimizer.step()
            loss_train = self.evaluate(loader_train)
            loss_test = self.evaluate(loader_test)
            self.loss_train_log.append(loss_train)
            self.loss_test_log.append(loss_test)
            # Save the best model
            if loss_test < best_loss:
                epoch_no_improvement = 0      
                best_loss = loss_test
                torch.save(self.state_dict(), 'best_model.pth')  # 🔐 Save best model
                print(f"Best model saved with loss: {best_loss:.4f} at {epoch+1}")
            else:
                epoch_no_improvement += 1
                if epoch_no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1} with no improvement.")
                    break
                
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {self.loss_train_log[-1]:.4f} | Test Loss: {self.loss_test_log[-1]:.4f}")

    def evaluate(self, loader):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.forward(x)
                loss = self.lossfn(y_pred, y)
                total_loss += loss.item()
        return total_loss / len(loader)
    
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
def collect_data(filename):
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

def determine_output(label):
    if label == 0:
        return "No detection"
    elif label == 1:
        return "Detected"
   
def predict(filename):
    adc_data = np.loadtxt(filename)
    combined_features = extract_features_from_adc(adc_data)
    # print(f"🔍 Extracted features from {filename}")
    feature_row = get_features_row(filename, combined_features, eval=True)
    # print(f"Feature row: {feature_row}")
    df = pd.DataFrame([feature_row])
    # print(df.values)
    X_raw = df.values.astype("float32")
    scaler: StandardScaler = joblib.load("./MLP_training_furthest_challenge/scaler.joblib")
    X_scaled = scaler.transform(X_raw)
    X_tensor = torch.from_numpy(X_scaled)
    X_tensor = X_tensor.to(device)
    input_dim = X_tensor.shape[1]
    model = DeepMLP(input_size=36, hidden_sizes=[256,128,64,32], output_size=8, p_dropout=0.3, device=device)
    # print(model.device)
    model.load_state_dict(torch.load("./MLP_training_furthest_challenge/best_model.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()  # <-- move to CPU first
    # print(f"Predictions shape: {preds.shape}")
    # print(f"Predictions: {preds}")
    label = determine_output(preds[0])
    print(f"Predicted classes: {label}")
    
# === Program Entry ===
if __name__ == "__main__":
    while True:
        print("\n=== MAIN MENU ===")
        print("1. Start Collecting Data")
        print("2. Start the prediction")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            collect_data(FILENAME)
        elif choice == '2':
            predict(FILENAME)
        elif choice == '3':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please select 1, 2.")
    
    # predict(FILENAME)