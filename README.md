# Fraud-detection-System

# Project Description
This project aims to build a classification model to accurately detect fraudulent credit card transactions, thereby reducing financial losses and improving security. The model is trained on the Credit Card Fraud Detection dataset from Kaggle, which contains simulated credit card transactions labeled as legitimate or fraudulent.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [License](#license)

# Installation
To run this project, you need to have Python installed along with the following libraries:
numpy
pandas
scikit-learn
matplotlib
You can install the required libraries using the following command:
```bash
pip install numpy pandas scikit-learn matplotlib
```

# Usage
1. Clone the repository:
```bash
git clone https://github.com/your_username/Fraud-Detection-System.git
cd Fraud-Detection-System
```

2. Run the jupyter notebook
```bash
jupyter notebook fdc.ipynb
```

# Project Structure
```bash
Fraud-Detection-System/
├── fdc.ipynb                          # Main Jupyter Notebook with the code
├── Dataset/
│   ├── creditcard.csv                 # Credit Card Fraud Detection dataset
├── README.md                          # Project README file
```

# Data Preprocessing
The dataset is loaded using pandas. Missing values are handled by filling them with the mean of the respective columns. Features are scaled using StandardScaler, and dimensionality reduction is performed using PCA.
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv('Dataset/creditcard.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(data.drop('Class', axis=1)), columns=data.drop('Class', axis=1).columns)

# Scale the features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# Perform PCA
pca = PCA(n_components=0.95)
X_reduced = pd.DataFrame(pca.fit_transform(X_scaled), columns=[f'PC{i+1}' for i in range(pca.n_components_)])
```

# Model Training
The data is split into training and testing sets. A Random Forest Classifier is trained on the preprocessed data.
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define features and target variable
X_final = X_reduced
y_final = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
```

# Model Evaluation
The model's performance is evaluated using the Precision-Recall Curve, F1 Score, Confusion Matrix, and Classification Report.
```python
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Calculate precision and recall
precision, recall, threshold = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
average_precision = average_precision_score(y_test, y_probs)

# Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc: .4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()

# Predict classes
y_pred = model.predict(X_test)

# Calculate F1 Score
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.4f}')

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
```

# Results
```bash
Confusion Matrix:
[[85303     4]
 [   31   105]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.96      0.77      0.86       136

    accuracy                           1.00     85443
   macro avg       0.98      0.89      0.93     85443
weighted avg       1.00      1.00      1.00     85443
```
# License
This README template includes all the pertinent information about your project, such as installation instructions, usage, project structure, data processing, model training, model evaluation, and details about the web application. It also includes sections for contributing and licensing, which are important for open-source projects.
