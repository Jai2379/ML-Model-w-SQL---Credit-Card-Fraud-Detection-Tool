# Credit Card Fraud Detection using Machine Learning

> **An automated, end-to-end machine learning pipeline to detect fraudulent credit card transactions with high precision.**

## 📌 Project Overview

**The Problem:** Credit card fraud costs the financial industry billions of dollars annually. Identifying malicious transactions among millions of genuine purchases is a complex "needle in a haystack" problem.

**The Goal:** Build a robust, real-time predictive model to identify fraudulent transactions, thereby minimizing financial risk and reducing false positives that inconvenience real customers.

**The Approach:** This project leverages Exploratory Data Analysis (EDA), statistical feature selection, and a Random Forest Classifier. Because fraud cases are extremely rare, we use **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset and ensure the model learns effectively.

***I developed this project to strengthen my data science skill set, recognizing its growing importance across industries. By working with Python, SQL, and machine learning techniques, I built a foundation in extracting actionable insights from data—skills that play a critical role in driving informed decisions, reducing risk, and improving business outcomes.***

### 🔍 Key Insights
- **Severe Class Imbalance:** Fraudulent transactions make up a tiny fraction of the total dataset. Without intervention, models become heavily biased toward predicting "Normal" transactions.
- **Distinct Behavioral Patterns:** Key anonymized PCA features (such as `V17`, `V14`, `V12`, `V10`, `V4`) show distinctly different mathematical distributions during a fraudulent event compared to normal spending behavior.
- **Transaction Amount Variance:** Fraudulent and normal transactions differ in their median amounts and overall spread. However, "stealth fraud" often mimics standard, low-amount daily purchases to avoid tripping simple threshold alarms.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, Imbalanced-learn (SMOTE)
- **Database:** SQLite3
- **Model Serialization:** Joblib

---

## 🚀 Installation

To set up this project locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Jai2379/ML-Model-w-SQL---Credit-Card-Fraud-Detection-Tool.git

# Navigate into the project directory
cd ML-Model-w-SQL---Credit-Card-Fraud-Detection-Tool

# Install the required dependencies
pip install -r requirements.txt

#Download the datasets for this model from the following:
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023
```

---

## 💻 Usage

The project is broken down into modular scripts for analysis, pipeline generation, and prediction.

**1. Run the core analysis and train the initial model:**
```bash
python analysis.py
```

**2. Trigger the automated pipeline for dynamic feature selection and retraining:**
```bash
python pipeline.py
```

**3. Test real-time predictions on dummy transaction data:**
```bash
python predict_fraud.py
```

---

## 📂 Project Structure

- `analysis.py`: Handles exploratory data analysis, SQLite integration, SMOTE balancing, and baseline model training.
- `pipeline.py`: An automated retraining pipeline (`FraudPipeline`) that dynamically selects top features (`SelectKBest`) and retrains the Random Forest model.
- `predict_fraud.py`: The live-scanning script that evaluates new transactions, outputting a probability score and fraud alert.
- `models/`: Directory housing the serialized, active machine learning models (e.g., `active_model.pkl`, `final_fraud_detector.pkl`).
- `fraud.ipynb`: A Jupyter Notebook containing in-depth EDA and data visualization.
- `creditcard.csv`: The raw transaction dataset (ignored in version control due to size constraints).
- `importer.py`: A utility script to upload new CSV datasets directly into the SQLite database for continuous testing and evaluation.

---

## 🌟 Features

- **Exploratory Data Analysis (EDA):** Deep dives into feature distributions and correlations to separate fraud from normal transactions.
- **Handling Class Imbalance:** Employs SMOTE to synthetically generate minority class examples, providing the AI with a balanced 50/50 split for fair training.
- **Dynamic Feature Selection:** Automatically extracts the top predictive red flags using ANOVA F-value statistical tests.
- **Real-Time Predictive Scoring:** Takes in live transaction dictionaries, applies the active model, and returns a detailed fraud probability score.

---

## 📈 Results

- **Initial Baseline Model:** Yielded sub-optimal accuracy and recall due to the overwhelming class imbalance (the model struggled to identify true fraud).
- **Balanced Model (SMOTE + Random Forest):** By synthetically balancing the data and hyperparameter tuning the Random Forest (e.g., capping `max_depth` to prevent overfitting), the model achieved significantly improved classification metrics, successfully isolating true fraud while minimizing false alarms.

---

## 🔮 Future Improvements

- **Algorithm Upgrades:** Experiment with advanced gradient boosting algorithms like XGBoost, LightGBM, or Deep Neural Networks to capture more complex, non-linear relationships.
- **Real-Time API Deployment:** Wrap the prediction logic in a FastAPI or Flask endpoint for seamless integration with a web or mobile banking application.
- **Advanced Feature Engineering:** Introduce time-series velocity features (e.g., number of transactions per hour) to detect rapid, successive fraud attempts.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page if you want to contribute. 
