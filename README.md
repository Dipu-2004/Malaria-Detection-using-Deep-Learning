# 🦠 Malaria Detection Using Deep Learning
### 📌 Project Overview

This project presents an automated Malaria Detection System using Deep Learning (MobileNetV2 Transfer Learning). The system classifies microscopic blood smear images as:

✅ Parasitized

✅ Uninfected

The trained model is deployed using a Flask web application, allowing users to upload images and receive real-time predictions.

### 🎯 Objective

To develop a robust and accurate deep learning model capable of detecting malaria parasites in cell images and deploy it as a web-based application.

### 🗂 Dataset Structure

Binary classification dataset

Balanced classes

Image augmentation applied

80% training / 20% validation split

### 🧠 Model Architecture

Base Model: MobileNetV2 (Pretrained on ImageNet)

Transfer Learning

Global Average Pooling

Dropout (0.5)

Dense Layer (128 neurons, ReLU)

Output Layer (1 neuron, Sigmoid)

### ⚙️ Technologies Used

Python

TensorFlow / Keras

MobileNetV2

Flask

NumPy

Matplotlib

Scikit-learn

### 🚀 Training Details

Image Size: 224 × 224

Batch Size: 32

Optimizer: Adam

Loss Function: Binary Crossentropy

Epochs: 20 (with Early Stopping)

Callbacks:

EarlyStopping

ReduceLROnPlateau

ModelCheckpoint

### 🌐 Web Application

The trained model is integrated into a Flask web app.

Features:

Upload microscopic cell image

Real-time prediction

Displays classification result

Lightweight deployment

### ▶️ How to Run the Project
1️⃣ Install Dependencies
pip install flask tensorflow numpy pillow matplotlib scikit-learn
2️⃣ Run Flask App
python app.py
3️⃣ Open in Browser
http://127.0.0.1:5000/

Upload an image and get prediction.

### 🏥 Real-World Applications

Automated malaria screening

Assistive diagnostic tool

Rural healthcare support systems

AI-based pathology assistance

### 🔮 Future Improvements

Grad-CAM visualization for explainability

Confidence score display

Deployment on cloud (Render / AWS / GCP)

Convert to REST API

Improve fine-tuning with deeper layer unfreezing

### 👨‍💻 Author

Dipankar
B.Tech Student
Deep Learning & AI Enthusiast
