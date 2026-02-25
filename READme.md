# ESP32 → Azure IoT Hub (DPS) Data Pipeline

This project demonstrates how an ESP32 device collects **current, voltage, and acceleration** data from a connected sensor module and sends it securely to **Microsoft Azure IoT Hub** using the **Device Provisioning Service (DPS)**.

---

## ✅ Features
- Collects real‑time electrical parameters:
  - Current
  - Voltage
  - Acceleration (3‑axis)
- Secure provisioning with Azure DPS
- MQTT‑based data transmission to Azure IoT Hub
- Configurable sampling rate
- Cloud integration ready for dashboards & analytics

---

## 📦 System Architecture
```
[Sensor Module] → [ESP32] → (MQTT via DPS) → [Azure IoT Hub] → [Stream Analytics] → [Database / Dashboard]
```

---

## 🛠️ Hardware Requirements
- ESP32 development board
- Current sensor (ACS712)
- Voltage sensor module
- Accelerometer (MPU6050 / ADXL345 or similar)
- Power supply & wiring

---

## 🧰 Software Requirements
- Arduino IDE / PlatformIO
- Azure IoT Hub
- Azure DPS
- MQTT libraries

---

## 🔗 Azure Setup Summary
1. Create IoT Hub
2. Create DPS instance
3. Link DPS to IoT Hub
4. Create enrollment group or individual enrollment
5. Get ID Scope, Registration ID, and Device Key
6. Configure ESP32 with credentials

---

## 🚀 Data Flow
Each reading packet follows a JSON format:
```json
{
  "deviceId": "esp32-demo",
  "current": 2.15,
  "voltage": 229.4,
  "accel": {
    "x": 0.13,
    "y": -0.02,
    "z": 9.77
  },
  "timestamp": "2025-11-08T12:40:30Z"
}
```

---

## 📊 Suggested Cloud Components
- Azure Stream Analytics → Route telemetry
- Azure Storage / SQL DB → Data persistence
- Power BI / Streamlit → Visualization
- Azure Functions → Event processing, anomaly detection

---

## 📁 Folder Structure
```
project-folder/
├─ firmware/      # ESP32 source code
├─ cloud/         # Azure configs & scripts
├─ docs/          # Architecture & notes
└─ README.md
```

---

## 📌 Future Enhancements
- Edge anomaly detection (TinyML)
- Over‑the‑air firmware updates
- Multi‑device provisioning
- Mobile app dashboard

---

## 👤 Author
Developer: Arun Raj


## Installation

### Prerequisites
- ESP32
- Azure Account
- Azure IoT Hub & DPS
- Python 3.12+
- pip
