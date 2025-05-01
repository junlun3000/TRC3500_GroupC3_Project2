import serial
import time
from datetime import datetime

# === CONFIGURATION ===
PORT = 'COM13'         # Change to your STM32 COM port
BAUDRATE = 1000000    # Must match STM32 UART config
TIMEOUT = 1           # seconds
LOG_TO_FILE = True

# === OUTPUT FILE (optional) ===
if LOG_TO_FILE:
    filename = f"adc_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logfile = open(filename, 'w')
    print(f"[INFO] Logging to {filename}")

# === OPEN SERIAL PORT ===
try:
    ser = serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT)
    print(f"[INFO] Connected to {PORT} at {BAUDRATE} baud.")
except serial.SerialException as e:
    print(f"[ERROR] Could not open serial port {PORT}: {e}")
    exit(1)

# === MAIN LOOP ===
try:
    while True:
        line = ser.readline().decode('utf-8').strip()

        if line:
            try:
                adc_value = int(line)
                print(f"ADC: {adc_value}")

                if LOG_TO_FILE:
                    logfile.write(f"{adc_value}\n")

            except ValueError:
                print(f"[WARN] Non-integer data received: {line}")

except KeyboardInterrupt:
    print("\n[INFO] Exiting...")

finally:
    ser.close()
    if LOG_TO_FILE:
        logfile.close()
    print("[INFO] Serial port closed. Bye 👋")
