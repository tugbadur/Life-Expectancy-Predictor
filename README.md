# Life Expectancy Prediction with Neural Networks

This project predicts **Life Expectancy** using a fully-connected neural network built with **TensorFlow/Keras**.  
The dataset includes health, economic, and demographic indicators for multiple countries, and the goal is to estimate life expectancy based on these features.

---

## 📌 Project Structure

life-expectancy-predictor/
│
├── data/
│ └── life_expectancy.csv
│
├── src/
│ └── train.py
│
├── requirements.txt
└── README.md

yaml
Kodu kopyala

---

## 🚀 Features

- Preprocessing with:
  - One-hot encoding for categorical variables
  - StandardScaler for numerical features
  - Train/Test split
- A feed-forward neural network with:
  - 64-unit ReLU hidden layer
  - 1-unit regression output
  - Adam optimizer
- Evaluation using **MSE** and **MAE**

---

## 🧠 Model Summary

- Input: dynamic based on feature count  
- Hidden layer: Dense(64, activation='relu')  
- Output layer: Dense(1)  
- Loss: Mean Squared Error  
- Metric: Mean Absolute Error  

---

## 📂 Dataset

The dataset used is `life_expectancy.csv`, which should contain:

- Country (removed during preprocessing)
- Health-related indicators
- Economic indicators
- Population data
- Life expectancy (label)

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
Run training:

bash
Kodu kopyala
python src/train.py
The script will:

Load and preprocess data

Train the neural network

Print training logs

Output MSE and MAE results

📈 Results
Training produces metrics similar to:

MSE: (varies by training)

MAE: (varies by training)

Results may differ due to randomness in weight initialization and train/test splitting.