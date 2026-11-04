# 📊 Executive Retail Intelligence & Demand Forecasting Suite

An end-to-end data engineering, machine learning, and interactive analytics dashboard built with **Python, Pandas, Scikit-Learn, and Streamlit**. 

This repository features a complete business intelligence pipeline—from automated data ingestion and exploratory visual analysis to unsupervised customer segmentation (RFM K-Means) and supervised sales forecasting (Random Forest Regressor)—wrapped in a production-grade, executive-styled Web Application.

---

## 🌟 Key Features

- **Automated Data Pipeline**: Cleaned, standardized, and feature-engineered raw transaction logs into analysis-ready data.
- **Executive BI Dashboard**: Interactive Streamlit application customized with light blue gradient aesthetics, custom metrics cards, and responsive sidebar filtering.
- **RFM Customer Segmentation**: Integrated K-Means Clustering on Recency, Frequency, and Monetary metrics to segment customer bases into actionable behavioral profiles.
- **Time-Series Sales Forecasting**: Random Forest Regression model utilizing lag features and rolling window statistics to predict monthly revenue trends.
- **Publication-Grade Visualizations**: Built with Seaborn and Matplotlib featuring clean chart styling, custom color palettes, and despined axes.

---

## 🏗️ Project Architecture

```text
Retail_Sales_Analysis/
├── Data_set/
│   ├── raw_retail_sales.csv       # Original dataset
│   └── cleaned_retail_sales.csv   # Post-pipeline processed data
├── src/
│   ├── data_cleaning.py           # Ingestion, imputation, and feature engineering logic
│   ├── visualize_sales.py         # Modular EDA and dashboard plotting scripts
│   ├── customer_segmentation.py   # RFM transformation and K-Means clustering model
│   └── sales_forecasting.py       # Time-series feature generation and Random Forest model
├── notebooks/
│   └── EDA.ipynb                  # Exploratory Data Analysis & experimentation notebook
├── app.py                         # Complete Streamlit Executive Dashboard Web Application
└── README.md                      # Comprehensive project documentation
```

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.9+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (`KMeans`, `RandomForestRegressor`, `StandardScaler`)
- **Data Visualization**: Matplotlib, Seaborn
- **Dashboard & UI**: Streamlit, Custom HTML/CSS

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9 or higher installed on your system.

### 2. Installation
Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/Jana-Abdelwahed/retail-sales-analysis.git
cd retail-sales-analysis

# Install required Python packages
python -m pip install -r requirements.txt
```

> *Note: If `requirements.txt` is not yet created, install directly via:*
> ```bash
> python -m pip install pandas numpy matplotlib seaborn scikit-learn streamlit
> ```

### 3. Running the Machine Learning Pipelines
You can execute individual modules from the project root:

```bash
# Run time-series sales forecasting
python src/sales_forecasting.py

# Run customer segmentation
python src/customer_segmentation.py
```

### 4. Launching the Web Dashboard
Start the interactive Streamlit dashboard locally:

```bash
python -m streamlit run app.py
```

Once launched, upload `cleaned_retail_sales.csv` using the sidebar control panel to populate the real-time analytics and cluster maps.

---

## 📈 Methodology Overview

### 1. Feature Engineering & RFM Analysis
Customer transaction logs are transformed into Recency (days since last purchase), Frequency (total transaction count), and Monetary (total spend) attributes. Log transformations ($ \log(x + 1) $) and standard scaling are applied prior to K-Means clustering to handle skewness and normalize feature variance.

### 2. Time-Series Forecasting
Monthly transaction aggregates use chronological train-test splits alongside custom time-lag variables (`Lag_1_Month`, `Lag_2_Month`, and `Rolling_Mean_3`) to train a `RandomForestRegressor` capable of capturing seasonal variations and moving trends without data leakage.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.