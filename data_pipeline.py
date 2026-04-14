#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import logging
import os

os.makedirs('logs', exist_ok=True)
logging.basicConfig(filename='logs/pipeline.log', level=logging.INFO)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)

def preprocess_data(data):
    
        data=data.copy()
        #Removal of duplicates 
        if data.duplicated().sum() > 0:
            data = data.drop_duplicates()

        # ===== MISSING VALUE HANDLING =====
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].median(), inplace=True)
                
        # Remove unrealistic ages
        data = data[data['person_age'] <= 100]

        # Employment length cannot exceed age
        data = data[data['person_emp_length'] <= data['person_age']]
        
        # Handling Outliers by capping 
        data['person_income'] = data['person_income'].clip(
            lower=data['person_income'].quantile(0.01),
            upper=data['person_income'].quantile(0.99))

        # ===== FEATURE ENGINEERING =====
        data['monthly_payment_est'] = data['loan_amnt'] / 12
        data['income_per_year_of_emp'] = data['person_income'] / (data['person_emp_length'] + 1)
        data['interest_income_ratio'] = data['loan_int_rate'] / (data['person_income'] + 1)
        data['credit_exp_ratio'] = data['cb_person_cred_hist_length'] / data['person_age']

        # ===== ENCODING =====
        data['loan_grade'] = data['loan_grade'].map({'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7})
        data['cb_person_default_on_file'] = data['cb_person_default_on_file'].map({'Y':1, 'N':0})
        data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent'], drop_first=True)

        return data


def run_data_pipeline():
    try:
        data = pd.read_csv('data/raw/credit_risk_dataset.csv')
        
        processed_data = preprocess_data(data)

        processed_data.to_csv('data/processed/data_preprocessed.csv', index=False)

        logging.info("Data pipeline completed successfully")

    except Exception as e:
        logging.error(f"Pipeline error: {e}")


if __name__ == "__main__":
    run_data_pipeline()


# In[ ]:




