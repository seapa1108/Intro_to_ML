# Intro_to_ML Course Repository

This repository contains implementations of fundamental machine learning models and a stacking model to improve performance.

## 📌 Overview
This repo is structured into two main parts:

**Basic Learners** – Implementation of five classic machine learning models from scratch (without using ML libraries like `scikit-learn`):
   1. Logistic Regression
   2. Multi-Layer Perceptron (MLP)
   3. Decision Tree
   4. k-Nearest Neighbors (KNN)
   5. Naïve Bayes

**Stacking Model** – An ensemble model that combines predictions from the basic learners to improve accuracy.

## 🛠️ Setup

### Requirements
Make sure you have the following dependencies installed:
```bash
pip install numpy pandas
```

### Running the Code
To train and evaluate each model, navigate to the corresponding directory and run:
```bash
python main.py
```

## 📊 Models & Implementation

### 🔹 Basic Learners
Each basic learner is implemented from scratch without using machine learning libraries like scikit-learn. They are trained individually and evaluated using performance metrics such as accuracy, fl score, and mcc score.

### 🔹 Stacking Model
The stacking model leverages the strengths of multiple models by:
1. Training each base model on the dataset.
2. Using their predictions as features for a Naïve Bayes meta-model.
3. Making final predictions based on the meta-model’s output.

## 📂 Repository Structure
```
/Machine_Learning
│── Logistic_Regression/   # Logistic Regression model
│   │── model.py           # Model implementation
│   │── main.py            # Training and evaluation script
│   │── preprocessor.py    # Data preprocessing
│   │── train_X.csv        # Training features
│   │── train_y.csv        # Training labels
│   │── test_X.csv         # Test features
│   │── test_y.csv         # Test labels
│
│── MLP/                   # Multi-Layer Perceptron model
│── Decision_Tree/         # Decision Tree model
│── KNN/                   # k-Nearest Neighbors model
│── Naive_Bayes/           # Naïve Bayes model
│── Stacking/              # Stacking model
│
│── README.md              # This file
```