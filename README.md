# House Rent Analysis and Prediction

## 📋 Project Overview
This project explores the Indian House Rent Dataset using Python and machine learning. It examines rental patterns across cities, studies how property features relate to rent, and compares classification models that predict whether a property’s rent is above the median rent for its city.

> **Important:** This project predicts a rent category, not an exact rental price. `High_Rent = 1` means the rent is above the median for that city; `High_Rent = 0` means it is at or below the city median.

## 🎯 Objectives
- Clean and prepare the house-rent dataset.
- Explore rent distributions and relationships between property features and rent.
- Use statistical tests to investigate differences across cities and categories.
- Engineer floor-related features and handle extreme values.
- Train and compare classification algorithms.
- Evaluate models using multiple performance metrics.

## ✨ Features
### 1. Data Cleaning and Feature Engineering
- Checks missing values and duplicate records.
- Parses floor information into `Floor_No` and `Total_Floors`.
- Removes selected columns with high cardinality or limited usefulness.
- Filters selected property records and identifies outliers using the IQR rule on log-transformed rent.
- Creates a city-relative `High_Rent` target.

### 2. Exploratory Data Analysis (EDA)
- Rent distribution before and after outlier filtering.
- Rent comparisons by city, BHK, and furnishing status.
- Relationship between property size and rent.
- Correlation heatmap and Pearson/Spearman correlation analysis.

### 3. Statistical Analysis
- One-way ANOVA and Kruskal–Wallis tests to compare rents across cities.
- Welch’s t-test for furnished versus unfurnished properties.
- Chi-square tests for associations between categorical features and `High_Rent`.

### 4. Machine Learning
Compares these classifiers:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)

KNN, Decision Tree, and SVM hyperparameters are tuned with `GridSearchCV` using stratified 5-fold cross-validation on the training set.

### 5. Model Evaluation
Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Log Loss
- Confusion matrices

## 🛠️ Technologies Used
Python, NumPy, Pandas, Matplotlib, Seaborn, SciPy, Statsmodels, and Scikit-learn.

## 📁 Repository Structure
```text
house-rent-analysis-machine-learning/
├── house_rent_analysis.py
├── README.md
├── requirements.txt
├── .gitignore
└── data/
    └── House_Rent_Dataset.csv   # add locally; do not commit unless permitted
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or later recommended
- pip
- The `House_Rent_Dataset.csv` dataset

### 1. Clone the repository
Replace `YOUR-USERNAME` with your GitHub username:

```bash
git clone https://github.com/YOUR-USERNAME/house-rent-analysis-machine-learning.git
cd house-rent-analysis-machine-learning
```

### 2. Create and activate a virtual environment (recommended)
Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Create a `data` folder in the project directory and place `House_Rent_Dataset.csv` inside it:

```text
data/House_Rent_Dataset.csv
```

Check the dataset's license and usage terms before redistributing it. The dataset is not included in this repository by default.

### 5. Run the project
```bash
python house_rent_analysis.py
```

The script prints data checks, statistical test results, and a model comparison table. It also displays exploratory charts and confusion matrices.

## 📊 Understanding the Target
The target is calculated separately for each city:
- **1 — High rent:** rent is greater than that city's median rent.
- **0 — Not high rent:** rent is equal to or lower than that city's median rent.

This makes the task a binary classification problem. It does not estimate an exact rent amount.

## ⚠️ Notes and Limitations
- Results depend on the dataset, filtering choices, and city representation.
- Outlier removal can discard valid high-rent properties and affect conclusions.
- City-relative labels compare properties within each city, not across the entire dataset.
- The current project is an analysis and classification workflow, not a deployed web application.
- If the dataset changes, rerun preprocessing and model evaluation.

## 🔮 Possible Future Improvements
- Add a separate regression model to estimate rental prices.
- Compare additional models and report the best cross-validated configuration.
- Save plots and model results to an output folder.
- Build a Streamlit web interface for interactive predictions.
- Add feature-importance or model-explanation visualizations.

## 📝 License
Add a license file if you want to specify how others may use, modify, and distribute your code. Check the dataset's license separately.

## 👤 Author
Add your name and GitHub profile here.

---
**Project:** House Rent Analysis and Prediction  
**Focus:** Data analysis, statistical testing, and machine learning classification
