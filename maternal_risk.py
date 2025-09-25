import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier

    xgb_available = True
except ImportError:
    xgb_available = False


# --------------------------
# Load and preprocess data
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Maternal Health Risk Data Set.csv")
    le = LabelEncoder()
    df["RiskLevel"] = le.fit_transform(df["RiskLevel"])
    return df, le


df, le = load_data()
X = df.drop("RiskLevel", axis=1)
y = df["RiskLevel"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# --------------------------
# Train models
# --------------------------
logit = LogisticRegression(max_iter=200)
logit.fit(x_train_scaled, y_train)
acc_logit = accuracy_score(y_test, logit.predict(x_test_scaled))

rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
acc_rf = accuracy_score(y_test, rf.predict(x_test))

if xgb_available:
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    xgb.fit(x_train, y_train)
    acc_xgb = accuracy_score(y_test, xgb.predict(x_test))
else:
    xgb, acc_xgb = None, 0

# Pick best model
models = {
    "Logistic Regression": (logit, acc_logit),
    "Random Forest": (rf, acc_rf),
    "XGBoost": (xgb, acc_xgb) if xgb_available else None
}
best_model_name, best_model, best_acc = max(
    [(name, model, acc) for name, val in models.items() if val is not None and val[1] is not None for model, acc in
     [val]], key=lambda x: x[2]   # compare by accuracy
)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Maternal Risk Prediction", page_icon="👩‍⚕️", layout="centered")

st.title("👩‍⚕️ Maternal Health Risk Prediction")
st.markdown("Enter patient details below to predict maternal health risk using **{}** (Accuracy: {:.2f}%)".format(
    best_model_name, best_acc * 100))
st.write("Note that the model below is build using a speci")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 10, 70, 25)
    systolic = st.number_input("Systolic BP", 80, 200, 120)
    diastolic = st.number_input("Diastolic BP", 50, 120, 80)

with col2:
    bs = st.number_input("Blood Sugar", 5.0, 20.0, 10.0, step=0.1)
    body_temp = st.number_input("Body Temperature (°F)", 95.0, 105.0, 98.0, step=0.1)
    hr = st.number_input("Heart Rate", 50, 150, 80)

# Predict button
if st.button("🔍 Predict Risk"):
    new_data = np.array([[age, systolic, diastolic, bs, body_temp, hr]])

    if best_model_name == "Logistic Regression":
        new_data_scaled = scaler.transform(new_data)
        pred = best_model.predict(new_data_scaled)
    else:
        pred = best_model.predict(new_data)

    risk_label = le.inverse_transform(pred)[0]

    # Show result in styled card
    st.markdown(f"""
        <div style="padding:20px; border-radius:15px; background:#f0f2f6; text-align:center;">
            <h2> 🩺 Predicted Risk Level: <span style="color:tomato;">{risk_label.upper()}</span></h2>
        </div>
    """, unsafe_allow_html=True)

# Show feature importance (for RF/XGB)
if best_model_name in ["Random Forest", "XGBoost"]:
    st.subheader("📊 Feature Importance")
    importance = best_model.feature_importances_
    fi_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)
    st.bar_chart(fi_df.set_index("Feature"))
