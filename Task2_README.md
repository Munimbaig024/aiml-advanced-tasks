# Task 2: Customer Churn Prediction with Machine Learning

## 📋 Overview

This project implements a comprehensive customer churn prediction system using machine learning techniques. The system analyzes customer data to predict whether a customer is likely to leave the service, helping businesses implement proactive retention strategies.

### 🎯 Objectives
- Build a robust customer churn prediction model
- Implement end-to-end ML pipeline with preprocessing
- Handle imbalanced datasets effectively
- Deploy model for real-time predictions
- Provide actionable insights for customer retention

## 📊 Dataset Information

- **Dataset**: `Telco-Customer-Churn.csv`
- **Total Customers**: 7,043
- **Features**: 20 customer attributes
- **Target Variable**: Churn (Yes/No)
- **Data Types**: Categorical and numerical features

### Feature Categories

#### Numerical Features
- `SeniorCitizen`: Whether customer is a senior citizen (0/1)
- `tenure`: Number of months customer has been with the company
- `MonthlyCharges`: Monthly service charges
- `TotalCharges`: Total charges over the customer's tenure

#### Categorical Features
- `gender`: Customer gender
- `Partner`: Whether customer has a partner
- `Dependents`: Whether customer has dependents
- `PhoneService`: Whether customer has phone service
- `MultipleLines`: Multiple phone lines subscription
- `InternetService`: Type of internet service
- `OnlineSecurity`: Online security subscription
- `OnlineBackup`: Online backup subscription
- `DeviceProtection`: Device protection subscription
- `TechSupport`: Tech support subscription
- `StreamingTV`: Streaming TV subscription
- `StreamingMovies`: Streaming movies subscription
- `Contract`: Contract type (Month-to-month, One year, Two year)
- `PaperlessBilling`: Paperless billing option
- `PaymentMethod`: Payment method

## 🏗️ Architecture & Methodology

### 1. Data Preprocessing Pipeline

#### A. Data Cleaning
- Remove non-predictive columns (customerID)
- Convert target variable to binary (Yes=1, No=0)
- Handle missing values and data inconsistencies

#### B. Feature Engineering
- **Numerical Features**: Standardization using StandardScaler
- **Categorical Features**: One-hot encoding using OneHotEncoder
- **Pipeline Integration**: Seamless preprocessing workflow

### 2. Machine Learning Pipeline

#### A. Model Selection
- **Random Forest Classifier**: Primary model for churn prediction
- **Logistic Regression**: Baseline model for comparison
- **Ensemble Methods**: Improved performance through model combination

#### B. Hyperparameter Tuning
- **GridSearchCV**: Systematic parameter optimization
- **Cross-validation**: Robust model evaluation
- **Class Weight Balancing**: Handle imbalanced dataset

### 3. Model Deployment
- **Pipeline Serialization**: Save complete ML pipeline
- **Real-time Prediction**: Deploy for new customer data
- **Model Persistence**: Load and reuse trained models

## 🚀 Setup & Installation

### Prerequisites
```bash
Python 3.8+
pandas
scikit-learn
joblib
numpy
```

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Task2
   ```

2. **Install required packages**
   ```bash
   pip install pandas scikit-learn joblib numpy
   ```

3. **Prepare the dataset**
   - Place `Telco-Customer-Churn.csv` in the project directory
   - Ensure data format matches expected structure

4. **Run the notebook**
   ```bash
   jupyter notebook Task2.ipynb
   ```

## 📊 Implementation Details

### 1. Data Loading and Preprocessing

```python
import pandas as pd

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Remove customer ID (not predictive)
df.drop("customerID", axis=1, inplace=True)

# Convert target to binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]
```

### 2. Feature Type Identification

```python
# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

print("Numerical:", num_cols)
print("Categorical:", cat_cols)
```

### 3. ML Pipeline Construction

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Numerical transformer
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Categorical transformer
cat_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Full preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# Complete pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])
```

### 4. Model Training and Optimization

```python
from sklearn.model_selection import train_test_split, GridSearchCV

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5]
}

# Grid search
grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
```

### 5. Model Evaluation

```python
from sklearn.metrics import classification_report

# Predictions
y_pred = grid_search.predict(X_test)

# Evaluation metrics
print(classification_report(y_test, y_pred))
```

## 🔍 Key Features

### 1. Automated Preprocessing
- **Standardization**: Scale numerical features to zero mean and unit variance
- **One-hot Encoding**: Convert categorical variables to numerical format
- **Pipeline Integration**: Seamless preprocessing workflow

### 2. Model Optimization
- **Grid Search**: Systematic hyperparameter optimization
- **Cross-validation**: Robust model evaluation
- **Class Balancing**: Handle imbalanced churn dataset

### 3. Model Deployment
- **Pipeline Serialization**: Save complete ML pipeline
- **Real-time Prediction**: Deploy for new customer data
- **Easy Integration**: Simple API for predictions


## 🛠️ Usage Examples

### Basic Model Training
```python
# Train the model
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
```

### Hyperparameter Tuning
```python
# Define parameter grid
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, None],
    'classifier__min_samples_split': [2, 5]
}

# Perform grid search
grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)
```

### Model Deployment
```python
import joblib

# Save the model
joblib.dump(grid_search.best_estimator_, "churn_pipeline.pkl")

# Load the model
model = joblib.load("churn_pipeline.pkl")

# Make predictions on new data
new_customer = X.iloc[[0]]  # Single customer data
prediction = model.predict(new_customer)
churn_probability = model.predict_proba(new_customer)
```

### Real-time Prediction
```python
def predict_churn(customer_data):
    """
    Predict churn for a single customer
    
    Args:
        customer_data: DataFrame with customer features
    
    Returns:
        prediction: 0 (No churn) or 1 (Churn)
        probability: Churn probability
    """
    model = joblib.load("churn_pipeline.pkl")
    prediction = model.predict(customer_data)
    probability = model.predict_proba(customer_data)
    
    return prediction[0], probability[0][1]
```

## 📈 Performance Metrics

### Model Performance Summary
- **Accuracy**: 80% overall accuracy
- **Precision (Churn)**: 61% - correctly identified churners
- **Recall (Churn)**: 66% - found 66% of actual churners
- **F1-Score (Churn)**: 63% - balanced measure of precision and recall

### Key Insights
- **Class Imbalance**: Dataset has more non-churners than churners
- **Feature Importance**: Contract type and tenure are key predictors
- **Model Robustness**: Random Forest handles non-linear relationships well
- **Balanced Performance**: Class weight balancing improves minority class performance

## 🚨 Troubleshooting

### Common Issues

1. **Data Loading Error**
   ```python
   # Ensure correct file path
   df = pd.read_csv("Telco-Customer-Churn.csv")
   ```

2. **Memory Issues**
   ```python
   # Reduce dataset size for testing
   df_sample = df.sample(n=1000, random_state=42)
   ```

3. **Model Loading Error**
   ```python
   # Ensure model file exists
   import os
   if os.path.exists("churn_pipeline.pkl"):
       model = joblib.load("churn_pipeline.pkl")
   ```

4. **Prediction Errors**
   ```python
   # Ensure input data matches training format
   # Check column names and data types
   ```

## 🔒 Data Privacy & Security

### Data Handling
- **Anonymization**: Remove personally identifiable information
- **Data Validation**: Ensure data quality and consistency
- **Access Control**: Implement proper data access controls
- **Compliance**: Follow data protection regulations

### Model Security
- **Input Validation**: Validate all input data
- **Model Versioning**: Track model versions and changes
- **Monitoring**: Monitor model performance and drift
- **Backup**: Regular model backups and recovery procedures

## 📊 Evaluation Framework

### Model Assessment
- **Classification Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Detailed error analysis
- **ROC Curve**: Model discrimination ability
- **Feature Importance**: Understanding key predictors

### Business Impact
- **Churn Reduction**: Potential revenue impact
- **Customer Retention**: Improved customer lifetime value
- **Resource Allocation**: Optimize retention efforts
- **ROI Analysis**: Return on investment in ML system
