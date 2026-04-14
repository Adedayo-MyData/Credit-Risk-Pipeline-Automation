#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os

def evaluate_model():

    os.makedirs('logs', exist_ok=True)

    xgb = joblib.load('models/xgb.pkl')
    x = joblib.load('data/processed/features.pkl')
    
    x_train = joblib.load('data/processed/x_train.pkl')
    y_train = joblib.load('data/processed/y_train.pkl')
    
    x_test = joblib.load('data/processed/x_test.pkl')
    y_test = joblib.load('data/processed/y_test.pkl')

    y_probs = xgb.predict_proba(x_test)[:,1]
    
    y_pred = xgb.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_probs)
    
# ===== CROSS VALIDATION =====
    scores = cross_val_score(xgb, x_train, y_train, cv=5, scoring='roc_auc')
    #print(scores)
    print('Average ROC-AUC Cross Validation Score on Train Dataset is: ', scores.mean())

    print("Current ROC-AUC on Test Dataset is:", auc)
    
     # ===== ROC CURVE =====
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("logs/roc_curve.png")
    plt.close()
    
     # ===== FEATURE IMPORTANCE =====
    importances = xgb.feature_importances_
    features = x.columns

    feat_imp = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    sns.barplot(x='Importance', y='Feature', data=feat_imp)
    plt.title("XGBoost Feature Importance")
    plt.savefig("logs/feature_importance.png")
    plt.close()


    # Save performance
    with open("logs/performance.txt", "a") as f:
        f.write(f"ROC-AUC: {auc}\n")

    if auc < 0.90:
        print("⚠️ WARNING: Model performance dropped!")


if __name__ == "__main__":
    evaluate_model()


# In[ ]:




