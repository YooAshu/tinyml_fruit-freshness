# TinyML Multimodal Food Freshness Monitoring System

A TinyML-based food freshness monitoring system that combines gas sensors and computer vision on an ESP32-CAM to classify banana freshness completely offline.

The project integrates embedded machine learning, image processing, and environmental sensing to estimate food quality in real time.

---
<img width="1448" height="1086" alt="bad" src="https://github.com/user-attachments/assets/9b67c746-3518-4bff-b42a-14ca9fa31e53" />

## Features

- Offline TinyML inference on ESP32
- ESP32-CAM image analysis
- MQ3 alcohol gas sensor
- MQ5 combustible gas sensor
- DHT22 temperature & humidity sensor
- XGBoost based classifier
- Multimodal feature fusion
- Four freshness stages
  - Fresh
  - Ripe
  - Overripe
  - Spoiled
- Ripeness Index estimation
- Edibility prediction
- Adulteration detection
- Days-left estimation

---

## Hardware

- ESP32-CAM
- ESP32 Dev Board
- MQ3 Gas Sensor
- MQ5 Gas Sensor
- DHT22 Temperature & Humidity Sensor
- Breadboard
- Jumper Wires
- USB Power Supply

---

## Software

- Arduino IDE
- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- XGBoost

---

## Dataset

The dataset consists of multimodal measurements collected from bananas across different freshness stages.

Features include

- Hue
- Saturation
- Value
- Texture
- Spot Ratio
- MQ3
- MQ5
- Temperature
- Humidity
- Storage Day

Labels include

- Freshness Status
- Edibility Status
- Adulteration Status

---

## Machine Learning Pipeline

Raw Sensor Data
↓

Image Feature Extraction

↓

Feature Engineering

↓

Feature Fusion

↓

Normalization

↓

XGBoost Training

↓

Model Export

↓

ESP32 Deployment

↓

Real-Time Prediction

---

## Data Flow
<img width="1024" height="1536" alt="flow_chart2" src="https://github.com/user-attachments/assets/89ae7669-3571-468f-9f5e-9e4b77f16018" />

---

## Hardware Connection
<img width="1536" height="1024" alt="hardware_connection2" src="https://github.com/user-attachments/assets/d4684e99-ce29-4a67-839b-762aba7803cb" />

---

## Repository Structure

```
FoodSpoilageESP32
│
├── dataset plots/          # Visualizations used in report
├── live testing images/    # Sample testing images
├── dataset.csv             # Training dataset
├── train.py                # Model training script
├── train_results.txt       # Training logs
├── FoodSpoilageESP32.ino   # ESP32 firmware
└── README.md
```

---

## Model Outputs

The deployed model predicts

- Freshness Class
- Ripeness Index
- Edibility Status
- Adulteration Detection
- Estimated Days Left

---

## Results

- TinyML deployment on ESP32
- Offline inference
- Real-time prediction
- Multimodal feature fusion
- Lightweight embedded implementation

---

## Future Improvements

- Support additional fruits
- Mobile application integration
- Larger dataset
- Automatic cloud synchronization
- Additional gas sensors
- Quantized deep learning models

---

## Author

**Asgar Hussain**
</br>
**Ankit Ray**
</br>
**Aryan**

## Supervisor
**Dr. Biswajit Ghosh

Bachelor of TEchnology (IT)

Techno Main Salt Lake
