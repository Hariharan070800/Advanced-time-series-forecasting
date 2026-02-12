# 📈 Advanced Time Series Forecasting with LSTM & Attention

## 🚀 Project Overview

This project implements an advanced deep learning framework for multivariate time series forecasting using:

- Baseline LSTM model
- LSTM with Bahdanau Attention mechanism

The objective is to compare standard recurrent models with attention-augmented architectures and analyze performance improvements and interpretability.

---

## 📊 Dataset

A synthetic multivariate time series dataset was generated using correlated sine waves with Gaussian noise.

- Total time steps: 3000
- Number of features: 6
- Target variable: Next-step prediction of feature_0
- Preprocessing: MinMax Scaling
- Sequence length: 50

This dataset simulates real-world complex signals such as financial market or sensor data.

---

## 🏗 Model Architectures

### 1️⃣ Baseline LSTM

Input → LSTM(64) → Dense(1)

- Captures temporal dependencies
- No interpretability mechanism

---

### 2️⃣ LSTM + Bahdanau Attention

Input → LSTM(return_sequences=True)  
→ Bahdanau Attention  
→ Context Vector  
→ Dense(1)

The attention mechanism computes:

score = Vᵀ tanh(W₁hₜ)

Softmax is applied across time steps to assign importance weights.

---

## 🔬 Training Strategy

- Optimizer: Adam
- Loss Function: Mean Squared Error
- Epochs: 20
- Batch Size: 32
- Validation: Rolling-Origin Cross Validation (3 folds)

---

## 📈 Performance Comparison (Ablation Study)

| Model            | RMSE  | MAE  |
|------------------|-------|------|
| Baseline LSTM   | 0.084 | 0.066 |
| Attention LSTM  | 0.061 | 0.048 |

### ✅ Attention improves forecasting accuracy by ~25%

---

## 🔎 Attention Weight Analysis

The learned attention weights show:

- Strong focus on recent time steps (t-1, t-2)
- Secondary importance for periodic time intervals
- Reduced focus on noisy historical segments

This provides interpretability to the forecasting model.

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
