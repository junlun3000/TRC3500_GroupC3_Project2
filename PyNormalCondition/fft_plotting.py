import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
    peaks, _ = find_peaks(adc_data, distance=2)  # distance=2 是为了避免检测太密
    num_peaks = len(peaks)

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

    # === Top 5 bins ===
    top5_indices = np.argsort(binned_energy)[-5:][::-1]
    top5_bin_energies = [binned_energy[idx] for idx in top5_indices]
    top5_bin_ranges = [f"{int(bin_centers[idx]-bin_size/2)}-{int(bin_centers[idx]+bin_size/2)}Hz" for idx in top5_indices]

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
        "Num_Peaks": num_peaks
    }

    for i in range(5):
        features[f"Top{i+1}_Bin_Energy"] = top5_bin_energies[i]
        features[f"Top{i+1}_Bin_Range"] = top5_bin_ranges[i]

        # 🔧 加这段，inside for-loop
        range_str = top5_bin_ranges[i].replace('Hz','')
        low_str, high_str = range_str.split('-')
        center_freq = (int(low_str) + int(high_str)) // 2
        features[f"Top{i+1}_Bin_Center"] = center_freq


    # （选项）也可以加回整个binned energy列表，如果后续要画图：
    features["Binned_Energy"] = binned_energy.tolist()

    return features

if __name__ == "__main__":
    adc_data = np.loadtxt('./AI_learning/data/adc_eraser_height30cm_distance10cm_5.data')

    combined_features = extract_features_from_adc(adc_data)

    flattened_features = {
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
        "Top1_Bin_Energy": combined_features["Top1_Bin_Energy"],
        "Top2_Bin_Energy": combined_features["Top2_Bin_Energy"],
        "Top3_Bin_Energy": combined_features["Top3_Bin_Energy"],
        "Top4_Bin_Energy": combined_features["Top4_Bin_Energy"],
        "Top5_Bin_Energy": combined_features["Top5_Bin_Energy"],
        "Top1_Bin_Center": combined_features["Top1_Bin_Center"],
        "Top2_Bin_Center": combined_features["Top2_Bin_Center"],
        "Top3_Bin_Center": combined_features["Top3_Bin_Center"],
        "Top4_Bin_Center": combined_features["Top4_Bin_Center"],
        "Top5_Bin_Center": combined_features["Top5_Bin_Center"]
    }

    df = pd.DataFrame([flattened_features])
    print(df)

    df.to_csv('./AI_learning/data/features_output_binned.csv', index=False)

    combined_features = extract_features_from_adc(adc_data)

    # 提取binned_energy
    binned_energy = np.array(combined_features["Binned_Energy"])
    bin_size = 100  # Hz
    Fs = 10000
    max_freq = Fs / 2
    bin_centers = np.arange(0, max_freq, bin_size) + bin_size / 2

    # 找 Top 3 bins
    top3_indices = np.argsort(binned_energy)[-5:][::-1]
    top3_bin_centers = bin_centers[top3_indices]
    top3_bin_energies = binned_energy[top3_indices]

    # 重新算 low/high band ratio (基于 binning后的)
    low_band_bins = bin_centers <= 600
    high_band_bins = bin_centers > 600
    low_band_energy_binned = np.sum(binned_energy[low_band_bins])
    high_band_energy_binned = np.sum(binned_energy[high_band_bins])
    total_energy_binned = low_band_energy_binned + high_band_energy_binned
    low_ratio_binned = low_band_energy_binned / total_energy_binned
    high_ratio_binned = high_band_energy_binned / total_energy_binned

    # 开始画图
    plt.figure(figsize=(14, 6))

    # 左边: Time domain
    plt.subplot(1, 2, 1)
    time_axis = np.arange(len(adc_data)) / Fs
    plt.plot(time_axis, adc_data)
    plt.title("Time Domain Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 右边: Binned frequency domain
    plt.subplot(1, 2, 2)
    bars = plt.bar(bin_centers, binned_energy, width=bin_size * 0.8, color='skyblue', edgecolor='black')

    # Top 3 bins 红色高亮
    for idx in range(5):
        plt.bar(top3_bin_centers[idx], top3_bin_energies[idx], width=bin_size * 0.8, color='red', edgecolor='black')
        plt.text(top3_bin_centers[idx], top3_bin_energies[idx]+np.max(binned_energy)*0.02,
                f"Top{idx+1}\n{top3_bin_centers[idx]:.0f}Hz", ha='center', color='red', fontsize=9)

    # 标注 Low Band
    plt.axvspan(0, 600, color='yellow', alpha=0.2)
    plt.text(250, np.max(binned_energy)*0.85,
            f"Low Band\n{low_ratio_binned*100:.1f}%", ha='center', fontsize=10,
            color='black', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    # 标注 High Band
    plt.axvspan(600, max_freq, color='lightgray', alpha=0.2)
    plt.text(3000, np.max(binned_energy)*0.85,
            f"High Band\n{high_ratio_binned*100:.1f}%", ha='center', fontsize=10,
            color='black', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    plt.title(f"Binned Energy per {bin_size}Hz (Frequency Domain)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Energy (Sum of Magnitude²)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()