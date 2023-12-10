import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import codecademylib3

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
print(transactions.info())

# How many fraudulent transactions?
fraud_num = transactions.isFraud.sum()
print(fraud_num)

# Summary statistics on amount column
print(transactions['amount'].describe())

# Create isPayment field
transactions['isPayment'] = 0
transactions['isPayment'][transactions['type'].isin(['PAYMENT','DEBIT'])] = 1

# Create isMovement field
transactions['isMovement'] = 0
transactions['isMovement'][transactions['type'].isin(['CASH_OUT','TRANSFER'])] = 1

# Create accountDiff field
transactions['accountDiff'] = abs(transactions['oldbalanceOrg'] - transactions['oldbalanceDest'])

# Create features and label variables
features = transactions[['amount', 'isPayment','isMovement','accountDiff']]
label = transactions['isFraud']

# Split dataset
x_train,x_test,y_train,y_test = train_test_split(features,label,test_size=0.3)

# Normalize the features variables
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Fit the model to the training data
model = LogisticRegression()
model.fit(x_train,y_train)

# Score the model on the training data
print(model.score(x_train,y_train))

# Score the model on the test data
print(model.score(x_test,y_test))

# Print the model coefficients
print(model.coef_)
print("amount feature was the most important, followed by isMovement. isPayment was the least important in predictin Fraud")
# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
your_transaction = np.array([1000.25, 0.0, 1.0, 1.99])

# Combine new transactions into a single array
sample_transactions = np.stack((transaction1, transaction2,transaction3,your_transaction))

# Normalize the new transactions
sample_transactions = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
print(model.predict(sample_transactions))

# Show probabilities on the new transactions
print(model.predict_proba(sample_transactions))