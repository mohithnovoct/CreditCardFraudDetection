import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('creditcard.csv')

# Data overview
print(data.head())
print(data.info())
print(data['Class'].value_counts())

# Check for missing values
print(data.isnull().sum())

# Data preprocessing
X = data.drop(['Class'], axis=1)
y = data['Class']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Isolation Forest model
isolation_forest = IsolationForest(contamination=0.01, random_state=42)
isolation_forest.fit(X_train)

# Predicting
y_pred_train = isolation_forest.predict(X_train)
y_pred_test = isolation_forest.predict(X_test)

# Converting -1 to 1 (fraud) and 1 to 0 (normal)
y_pred_train = [1 if x == -1 else 0 for x in y_pred_train]
y_pred_test = [1 if x == -1 else 0 for x in y_pred_test]

# Evaluation
print('Confusion Matrix - Test Set')
print(confusion_matrix(y_test, y_pred_test))

print('Classification Report - Test Set')
print(classification_report(y_test, y_pred_test))

# Visualizing the distribution of fraud vs normal transactions
plt.figure(figsize=(10, 6))
sns.countplot(x='Class', data=data)
plt.title('Distribution of Fraud vs Normal Transactions')
plt.show()
