#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import joblib
import os

def run_prediction():

    os.makedirs('data/processed', exist_ok=True)
    
    # LOAD
    xgb = joblib.load('models/xgb.pkl')

    x_test = joblib.load('data/processed/x_test.pkl')

    # Prediction
    y_pred = xgb.predict(x_test)
    
    # Probabilities
    y_probs = xgb.predict_proba(x_test)[:,1]

    x_test['prediction'] = y_pred
    x_test['probability'] = y_probs

    x_test.to_csv('data/processed/predictions.csv', index=False)

    print("Predictions saved")


if __name__ == "__main__":
    run_prediction()


# In[ ]:




