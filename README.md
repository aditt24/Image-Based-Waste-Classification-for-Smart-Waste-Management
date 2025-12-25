# ♻️ Waste Classification Using Deep Learning  
**Image-Based Waste Classification for Smart Waste Management**

**Author:** Putra  
**Role:** Data Scientist / Machine Learning Engineer  

**Main Tools:**  
Python · TensorFlow · Keras · Streamlit · Optuna · Computer Vision

---

## 📌 Project Overview
This project focuses on building an **image-based waste classification system** using Deep Learning to support **smart waste management**.  
The trained model is deployed as a **Streamlit web application**, allowing users to upload waste images and receive real-time classification results.

---

## 🧩 Problem Statement
- Waste is often mixed, making recycling inefficient  
- Manual waste sorting is time-consuming and error-prone  
- An automated AI-based solution is needed to assist waste classification  

---

## 🎯 Project Objectives
- Build an image-based waste classification model  
- Classify waste into **three categories**:
  - Compost (Organic)
  - Recyclable
  - Non-Recyclable
- Deploy the trained model into an interactive **web application**

---

## 📊 Dataset
**Source:** Kaggle  
🔗 https://www.kaggle.com/datasets/phenomsg/waste-classification/data  

**Original Dataset Classes:**
- Organic  
- Recyclable  
- Non-Recyclable  
- Hazardous  

**Final Classes Used (3 Classes):**
- Compost (Organic)  
- Recyclable  
- Non-Recyclable  

⚠️ **Note:**  
The dataset (~1.1 GB) is **not included in this repository** due to GitHub file size limitations.  
Please download it directly from Kaggle using the link above.

---

## 🧹 Data Preparation
- Class mapping applied:
  - Organic → Compost  
  - Recyclable → Recyclable  
  - Non-Recyclable + Hazardous → Non-Recyclable  
- Data split:
  - Training: 80%
  - Validation: 20%
- Image resizing to **224 × 224**
- Directory-based data loading

---

## 🔄 Data Augmentation
To improve generalization, the following augmentations were applied:
- Rotation
- Zoom
- Width & height shift
- Horizontal flip
- Brightness adjustment

---

## 🧠 Model Architecture
- **Backbone:** EfficientNetB0 (pretrained on ImageNet)
- Global Average Pooling
- Fully Connected Layer (Dense)
- Batch Normalization
- Dropout (regularization)
- Softmax Output (3 classes)

This architecture balances performance and efficiency.

---

## ⚙️ Hyperparameter Optimization
- Optimization tool: **Optuna**
- Tuned parameters:
  - Learning rate
  - Dropout rate
  - Number of dense units
- Objective: **Maximize validation accuracy**

---

## 🏋️ Training Strategy
- **Stage 1:** Feature extraction  
  - Backbone frozen  
- **Stage 2:** Fine-tuning  
  - Last 20 layers unfrozen  
- Training techniques:
  - EarlyStopping
  - ReduceLROnPlateau
  - Label smoothing (0.1)

---

## 📈 Model Evaluation
**Validation Results:**
- Accuracy: **83%**
- Macro F1-score: **0.81**
- Balanced performance across all classes

These results indicate the model generalizes well and performs consistently.

---

## 🔍 Confusion Matrix Insights
- Non-recyclable class shows the most stable predictions  
- Some confusion exists between biodegradable and non-recyclable waste due to visual similarity  
- Overall classification performance is reliable

---

## 🌐 Deployment: Streamlit Web App
The trained model is deployed using **Streamlit** as an interactive web application.

**Application Workflow:**
1. Upload one or multiple waste images  
2. Model processes the images  
3. Prediction results are displayed with confidence scores  

This makes the project practical and user-friendly.

