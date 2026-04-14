#!/usr/bin/env python
# coding: utf-8

# In[2]:


import logging
import sys
import os

BASE_DIR = os.getcwd()
sys.path.append(os.path.join(BASE_DIR, 'scripts'))

from data_pipeline import run_data_pipeline
from train_model import train_model
from predict import run_prediction
from evaluation import evaluate_model

# Ensure required folders exist
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)


# Setup logging
logging.basicConfig(
    filename='logs/main_pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_full_pipeline():

    try:
        print("🚀 Starting Full Credit Risk Pipeline...")
        logging.info("Pipeline started")

        # STEP 1: Data preprocessing
        print("📊 Running Data Pipeline...")
        run_data_pipeline()
        logging.info("Data pipeline completed")

        # STEP 2: Model training
        print("🤖 Training Model...")
        train_model()
        logging.info("Model Training completed")

        # STEP 3: Prediction
        print("🔮 Running Predictions...")
        run_prediction()
        logging.info("Predictions completed")

        # STEP 4: Evaluation
        print("📈 Evaluating Model...")
        evaluate_model()
        logging.info("Evaluation completed")

        print("✅ Pipeline completed successfully!")
        logging.info("Pipeline completed successfully")

    except Exception as e:
        print("❌ Pipeline failed:", e)
        logging.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    run_full_pipeline()


# In[ ]:




