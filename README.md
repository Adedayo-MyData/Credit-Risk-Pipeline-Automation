# 💳 Credit Risk Pipeline Automation

## 📌 Project Overview
This project implements an **end-to-end automated machine learning pipeline** for **credit risk prediction**.  
It streamlines the process from data ingestion to model evaluation and prediction, enabling consistent and scalable decision-making.

The system is designed to:
- Continuously process new data  
- Retrain models  
- Generate predictions  
- Monitor model performance over time  

---

## 🚀 Key Features
- ✅ Automated data processing pipeline  
- ✅ Modular ML workflow (train, predict, evaluate)  
- ✅ Model performance tracking  
- ✅ Scheduled execution using Windows Task Scheduler  
- ✅ Scalable and reusable pipeline architecture  

---

## 🔄 Pipeline Workflow

The pipeline follows a structured workflow:

1. [Data Pipeline](data_pipeline.py)
   - Loads raw credit data  
   - Handles missing values  
   - Performs feature engineering  
   - Outputs clean dataset  

2. [Model Training](train_model.py)
   - Trains machine learning model  
   - Saves trained model for reuse  

3. [Prediction](predict.py)
   - Loads trained model  
   - Generates predictions on new/unseen data  

4. [Evaluation](evaluation.py)
   - Evaluates model performance  
   - Tracks key metrics (Accuracy, Precision, Recall, etc.)  

5. [Pipeline Orchestration](main_pipeline.py)
   - Runs all steps sequentially  
   - Ensures smooth automation of the workflow  

---

## ⏰ Automation & Scheduling

The pipeline is automated using **Windows Task Scheduler**:

- Runs **weekly**
- Retrains the model with updated data  
- Generates fresh predictions  
- Monitors performance trends  

👉 This ensures:
- Model stays up-to-date  
- Performance degradation is detected early  

---

## 📊 Use Case

This system helps financial institutions:

- Assess **creditworthiness of customers**
- Reduce **default risk**
- Make **data-driven lending decisions**
- Automate risk monitoring processes  

---

## 🧠 Model Insights

The model predicts:
- **Probability of loan default**

This enables:
- Risk-based customer segmentation  
- Better loan approval strategies  

---

## ⚙️ How to Run the Pipeline

### Run Full Pipeline
```bash
python main_pipeline.py
```

---

## 🔮 Future Improvements
- Deploy as a web API (FastAPI / Flask)
- Integrate real-time data streaming
- Add dashboard for monitoring (Power BI / Streamlit)
- Implement model versioning (MLflow)

---

## 👤 Author

Adedayo Adebayo
