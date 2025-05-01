# Piezoelectric Sensor System for Pin Drop Detection

This project implements a vibration-based classification system using a piezoelectric sensor and STM32 microcontroller to detect and distinguish objects (coin vs. eraser) dropped under varying conditions.

## 👥 Team (Group C3)
- Chai Jun Lun (33454930)  
- Louis Aristio (33361126)  
- Nandhana Alif Kusumabrata (33404526)  
- Vincent Law Yun Kae (32840152)  
- William Melvern Yang (33291128)  

## 🛠 Project Overview

The system captures mechanical vibrations using a piezoelectric sensor. These signals are amplified, digitized using STM32's ADC, and analyzed in Python for classification via AI.

### Key Features:
- Signal conditioning using TLV9054 op-amp  
- Real-time data acquisition via UART (1 Mbit/s)  
- Feature extraction (RMS, Peak, ZCR, FFT bins, etc.)  
- AI model trained to classify object type based on vibration

## 🔬 Test Setup

- **Objects**: 50 sen coin, eraser  
- **Heights**: 10 cm, 30 cm  
- **Distances from sensor**: 10 cm, 30 cm  
- **Classes**: 8 total (2×2×2 combinations)  
- **Samples per class**: 100 for training

## 📊 Signal Features Extracted

- Time-domain: RMS, Peak Amplitude, ZCR, Num Peaks, IQR Ratio, etc.  
- Frequency-domain: Spectral Centroid, Low/High Band Ratios, Top FFT Bin Energies  
- Output used to train a machine learning classifier

## 💡 Methodology

1. Object dropped after triggering via GUI  
2. STM32 samples ADC data at 10–12 kHz  
3. Data sent to Python via serial  
4. FFT and feature extraction performed  
5. AI model classifies object from signal

## 🧪 Tools & Equipment

- STM32 microcontroller  
- Piezoelectric sensor  
- ADALM1000 + Scopy (for waveform verification)  
- Python (NumPy, SciPy, scikit-learn)

## 📎 References

- [1] Week 3: Signal Conditioning and Transformation  
- [2] Project 2 – Briefing and Rubric  

---

> **Note**: Grammar and formatting reviewed using OpenAI tools.
