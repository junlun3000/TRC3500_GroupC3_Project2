import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, hilbert
from scipy.stats import skew, kurtosis

def extract_features_from_adc(adc_data, Fs=10000, bin_size=100):
    T = 1 / Fs
    num_samples = len(adc_data)

    # === Time Domain Features ===
    rms = np.sqrt(np.mean(adc_data ** 2))
    peak_amp = np.max(np.abs(adc_data))
    adc_zero_mean = adc_data - np.mean(adc_data)
    zero_crossings = np.where(np.diff(np.signbit(adc_zero_mean)))[0]
    zcr = len(zero_crossings) / (num_samples * T)
    # --- Number of Peaks ---
    peaks, _ = find_peaks(adc_data, prominence=20)
    num_peaks = len(peaks)
    
    # --- New: Envelope-based features via Hilbert transform ---
    analytic_signal = hilbert(adc_data)
    envelope = np.abs(analytic_signal)
    env_rms = np.sqrt(np.mean(envelope**2))       # Envelope RMS
    env_peak = np.max(envelope)                   # Envelope Peak


    # === Amplitude Spread Features ===
    amp_max = np.max(adc_data)
    amp_min = np.min(adc_data)
    amp_mean = np.mean(adc_data)
    amp_range = amp_max - amp_min
    spread_ratio = amp_range / amp_mean

    q1 = np.percentile(adc_data, 25)
    q3 = np.percentile(adc_data, 75)
    iqr = q3 - q1
    iqr_ratio = iqr / np.median(adc_data)
    
     # --- New: Peak-to-peak interval statistics ---
    if num_peaks > 1:
        peak_times = peaks / Fs
        intervals = np.diff(peak_times)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
    else:
        mean_interval = 0.0
        std_interval = 0.0

    # --- New: Shape descriptors ---
    waveform_skew = skew(adc_data)
    waveform_kurt = kurtosis(adc_data)

    # === Frequency Domain Features ===
    fft_result = np.fft.fft(adc_zero_mean)
    fft_freqs = np.fft.fftfreq(num_samples, d=1/Fs)
    fft_magnitude = np.abs(fft_result)

    positive_freqs = fft_freqs[:num_samples // 2]
    positive_magnitude = fft_magnitude[:num_samples // 2]
    

    # === REMOVE strongest peak before binning ===
    max_idx = np.argmax(positive_magnitude)
    positive_magnitude[max_idx] = 0

    # === Binning ===
    max_freq = Fs / 2
    num_bins = int(np.ceil(max_freq / bin_size))
    binned_energy = np.zeros(num_bins)
    bin_centers = np.arange(0, max_freq, bin_size) + bin_size / 2

    for i in range(num_bins):
        bin_mask = (positive_freqs >= i * bin_size) & (positive_freqs < (i + 1) * bin_size)
        binned_energy[i] = np.sum(positive_magnitude[bin_mask] ** 2)

    # === Top 10 bins ===
    top10_indices = np.argsort(binned_energy)[-10:][::-1]
    top10_bin_energies = [binned_energy[idx] for idx in top10_indices]
    top10_bin_ranges = [f"{int(bin_centers[idx]-bin_size/2)}-{int(bin_centers[idx]+bin_size/2)}Hz" for idx in top10_indices]

    # === Spectral centroid ===
    spectral_centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)

    # === Energy ratios ===
    low_band_mask = (positive_freqs >= 0) & (positive_freqs <= 600)
    high_band_mask = (positive_freqs > 600)
    low_band_energy = np.sum(positive_magnitude[low_band_mask] ** 2)
    high_band_energy = np.sum(positive_magnitude[high_band_mask] ** 2)
    total_energy = low_band_energy + high_band_energy
    low_ratio = low_band_energy / total_energy
    high_ratio = high_band_energy / total_energy

    # === Assemble features ===
    features = {
        "RMS": rms,
        "Peak_Amplitude": peak_amp,
        "ZCR": zcr,
        "Amplitude_Range": amp_range,
        "Spread_Ratio": spread_ratio,
        "IQR_Ratio": iqr_ratio,
        "Spectral_Centroid_Hz": spectral_centroid,
        "Low_Band_Ratio": low_ratio,
        "High_Band_Ratio": high_ratio,
        "Num_Peaks": num_peaks,
        "Env_RMS": env_rms,
        "Env_Peak": env_peak,
        "Mean_Interval": mean_interval,
        "Std_Interval": std_interval,
        "Waveform_Skew": waveform_skew,
        "Waveform_Kurt": waveform_kurt,
    }

    for i in range(10):
        features[f"Top{i+1}_Bin_Energy"] = top10_bin_energies[i]
        features[f"Top{i+1}_Bin_Range"] = top10_bin_ranges[i]

        # Center frequency
        range_str = top10_bin_ranges[i].replace('Hz','')
        low_str, high_str = range_str.split('-')
        center_freq = (int(low_str) + int(high_str)) // 2
        features[f"Top{i+1}_Bin_Center"] = center_freq


    # （选项）也可以加回整个binned energy列表，如果后续要画图：
    features["Binned_Energy"] = binned_energy.tolist()

    return features

def determine_label(object_name):
    if object_name == 'env':
        return 0
    elif object_name == 'eraser':
        return 1
    else:
        raise ValueError("Unknown combination!")
    
def get_features_row(file_path, combined_features, eval=False):
    # ==== 自己从路径推断 object, height, distance ====
    if not eval:
        parts = file_path.lower().split(os.sep)
        if 'env' in parts:
            obj = 'env'
        elif 'eraser' in parts:
            obj = 'eraser'
        else:
            return None
        
        label = determine_label(obj)

    # ==== 准备一行 feature ====
    feature_row = {
        "RMS": combined_features["RMS"],
        "Peak_Amplitude": combined_features["Peak_Amplitude"],
        "Num_Peaks": combined_features["Num_Peaks"],
        "ZCR": combined_features["ZCR"],
        "Amplitude_Range": combined_features["Amplitude_Range"],
        "Spread_Ratio": combined_features["Spread_Ratio"],
        "IQR_Ratio": combined_features["IQR_Ratio"],
        "Spectral_Centroid_Hz": combined_features["Spectral_Centroid_Hz"],
        "Low_Band_Ratio": combined_features["Low_Band_Ratio"],
        "High_Band_Ratio": combined_features["High_Band_Ratio"],
        "Env_RMS": combined_features["Env_RMS"],
        "Env_Peak": combined_features["Env_Peak"],
        "Mean_Interval": combined_features["Mean_Interval"],
        "Std_Interval": combined_features["Std_Interval"],
        "Waveform_Skew": combined_features["Waveform_Skew"],
        "Waveform_Kurt": combined_features["Waveform_Kurt"],
    }
    for i in range(10):
        feature_row[f"Top{i+1}_Bin_Energy"] = combined_features[f"Top{i+1}_Bin_Energy"]
        feature_row[f"Top{i+1}_Bin_Center"] = combined_features[f"Top{i+1}_Bin_Center"]

    if not eval:
        feature_row["Label"] = label
    return feature_row
    

if __name__ == "__main__":
    dataset_dir = './data_furthest_challenge'   # 顶层目录
    output_csv_path = './features_furthest_challenge.csv'  # 输出路径
    all_feature_rows = []

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.data'):
                file_path = os.path.join(root, file)
                adc_data = np.loadtxt(file_path)

                combined_features = extract_features_from_adc(adc_data)
                # print(f"🔍 Extracted features from {file_path}")
                # print(combined_features)

                feature_row = get_features_row(file_path, combined_features)
                if feature_row is None:
                    continue
                
                all_feature_rows.append(feature_row)

    # 全部存进一个大的 DataFrame
    df_all = pd.DataFrame(all_feature_rows)
    df_all.to_csv(output_csv_path, index=False)
    print(f"✅ Total {len(df_all)} samples saved to {output_csv_path}")