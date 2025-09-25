# 👩‍⚕️ Maternal Health Risk Prediction

![ML Badge](https://img.shields.io/badge/Machine%20Learning-Healthcare-blue?style=for-the-badge)  
A **machine learning-powered Streamlit app** for predicting **maternal health risk levels** (Low, Medium, High) based on clinical data.  

---

## 📌 Overview
Maternal health complications are a major concern worldwide. Early prediction of maternal risk can save lives.  

This project uses **machine learning algorithms** (Logistic Regression, Random Forest, XGBoost) trained on the *Maternal Health Risk Dataset* to classify patients into risk categories.  

The app allows healthcare professionals to:  
- Input patient details (Age, BP, BS, Body Temp, Heart Rate).  
- Get instant predictions of **maternal risk level**.  
- View model performance and feature importance.  

---

## 🚀 Features
- ✅ **Real-time Prediction** with patient input  
- ✅ **Machine Learning Models** (Logistic Regression, Random Forest, XGBoost)  
- ✅ **Feature Importance Visualization** (for RF/XGB)  
- ✅ **Streamlit UI** with interactive form and styled prediction output  

---

## 📂 Project Structure
```

maternal-health-risk/
│── app.py                        # Main Streamlit app (one page)
│── Maternal Health Risk Data Set.csv
│── requirements.txt
│── README.md

````

---
## ⚙️ Installation & Setup

1. **Clone the repository**
   
```bash
   git clone https://github.com/your-username/maternal-health-risk.git
   cd maternal-health-risk
````

2. **Create virtual environment**

```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
   pip install -r requirements.txt
```

4. **Run the app**

```bash
   streamlit run app.py
```

---

## 📊 Dataset

The dataset contains **1,014 records** with clinical features:

* `Age`
* `SystolicBP` (Systolic Blood Pressure)
* `DiastolicBP` (Diastolic Blood Pressure)
* `BS` (Blood Sugar)
* `BodyTemp` (Body Temperature in °F)
* `HeartRate`
* `RiskLevel` (Target: Low, Medium, High risk)

---

## 🧠 Machine Learning Workflow

1. **Preprocessing**: Label encoding, scaling, train-test split
2. **Model Training**: Logistic Regression, Random Forest, XGBoost
3. **Evaluation**: Accuracy, confusion matrix, classification report
4. **Prediction**: User enters details → Model outputs **risk level**
5. **Interpretability**: Feature importance plot for tree-based models

---

## 📸 App Preview

### 🩺 Prediction Page

![App Screenshot](App_UI.png?text=Maternal+Risk+Prediction+App)


## 📌 Future Improvements

* Use real clinical dataset 
* Use **SHAP/LIME** for explainability
* Extend dataset with more clinical features
* Deploy on hospital systems with secure APIs

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo
2. Create a branch (`git checkout -b feature-name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push to branch (`git push origin feature-name`)
5. Open a Pull Request

---


## 👨‍💻 Author

* **Abdulrazaq Isah Dikko** – [GitHub](https://github.com/Sir-ABD)
✨ If you like this project, please **star ⭐ the repo** to support!
