#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score
sns.set()

# Ensure folders exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

def train_model():

    # ===== LOAD DATA =====
    data_preprocessed = pd.read_csv('data/processed/data_preprocessed.csv')
    
    data = data_preprocessed.copy()
   

    # ===== SPLIT =====
    x=data.drop('loan_status',axis=1)
    y=data['loan_status']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 20)

    # ===== MODEL =====
   
    #XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight= (len(y_train[y_train==0]) / len(y_train[y_train==1])),
        random_state=42
    )

    xgb.fit(x_train, y_train)
    
    # ===== SAVE MODEL =====
    joblib.dump(xgb, 'models/xgb.pkl')
    
    # Save test data input and output
    
    joblib.dump(x_train, 'data/processed/x_train.pkl')
    joblib.dump(x_test, 'data/processed/x_test.pkl')
    joblib.dump(y_train, 'data/processed/y_train.pkl')
    joblib.dump(y_test, 'data/processed/y_test.pkl')
    joblib.dump(x, 'data/processed/features.pkl')
   

    print("Model training completed")

    return 


if __name__ == "__main__":
    train_model()


# In[ ]:




