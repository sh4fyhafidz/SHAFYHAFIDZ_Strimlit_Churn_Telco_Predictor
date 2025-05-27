import streamlit as st
import pandas as pd
import pickle
import requests
import io
import numpy as np
from sklearn.exceptions import NotFittedError

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Telco Customer Churn Prediction (Batch & Manual)")

# Load Model Pipeline
model_url = "https://raw.githubusercontent.com/sh4fyhafidz/FINPRO_GAMMA_JCDS0508_STRIMLIT/main/FINPRO_GAMMA_JCDS0508_Strimlit/Model_Telco_Company.sav"

@st.cache_data(show_spinner="Loading prediction model...")
def load_model_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        model = pickle.load(io.BytesIO(response.content))
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

model = load_model_from_url(model_url)

# Sidebar Batch Upload
st.sidebar.header("Batch Prediction")
uploaded_file = st.sidebar.file_uploader("📤 Upload CSV file", type=["csv"],
                                         help="Upload file with customer data in CSV format")
min_tenure = st.sidebar.number_input("🔢 Minimal Tenure (bulan):", min_value=0, value=10,
                                     help="Filter customers with tenure greater than this value")

# Sidebar Manual Input
st.sidebar.header("Manual Input Prediction")

def clean_and_validate_data(df):
    df = df.copy()
    # Drop cols if exist
    df = df.drop(columns=['CustomerID', 'Churn'], errors='ignore')

    required_columns = {
        'numeric': ['Tenure', 'MonthlyCharges', 'TotalCharges'],
        'categorical': ['Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                       'PaperlessBilling', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaymentMethod']
    }

    missing_cols = [col for col in required_columns['numeric'] + required_columns['categorical']
                    if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        st.stop()

    # Clean numeric columns
    for col in required_columns['numeric']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        if col == 'Tenure':
            df[col] = df[col].astype(int)

    # Clean categorical columns
    cat_mappings = {
        'Gender': {'Male': 'Male', 'Female': 'Female', 'M': 'Male', 'F': 'Female'},
        'SeniorCitizen': {'Yes': 'Yes', 'No': 'No', 'Y': 'Yes', 'N': 'No', '1': 'Yes', '0': 'No'},
        'Partner': {'Yes': 'Yes', 'No': 'No', 'Y': 'Yes', 'N': 'No'},
        'Dependents': {'Yes': 'Yes', 'No': 'No', 'Y': 'Yes', 'N': 'No'},
        'PhoneService': {'Yes': 'Yes', 'No': 'No', 'Y': 'Yes', 'N': 'No'},
        'PaperlessBilling': {'Yes': 'Yes', 'No': 'No'},
        'MultipleLines': {'Yes': 'Yes', 'No': 'No', 'No phone service': 'No phone service'},
        'InternetService': {'DSL': 'DSL', 'Fiber optic': 'Fiber optic', 'No': 'No'},
        'OnlineSecurity': {'No internet service': 'No internet service', 'Yes': 'Yes', 'No': 'No'},
        'OnlineBackup': {'No internet service': 'No internet service', 'Yes': 'Yes', 'No': 'No'},
        'DeviceProtection': {'No internet service': 'No internet service', 'Yes': 'Yes', 'No': 'No'},
        'TechSupport': {'No internet service': 'No internet service', 'Yes': 'Yes', 'No': 'No'},
        'StreamingTV': {'No internet service': 'No internet service', 'Yes': 'Yes', 'No': 'No'},
        'StreamingMovies': {'No internet service': 'No internet service', 'Yes': 'Yes', 'No': 'No'},
        'Contract': {'Month-to-month': 'Month-to-month', 'One year': 'One year', 'Two year': 'Two year',
                     'Monthly': 'Month-to-month', 'Yearly': 'One year'},
        'PaymentMethod': {'Mailed check': 'Mailed check', 'Credit card (automatic)': 'Credit card (automatic)',
                          'Bank transfer (automatic)': 'Bank transfer (automatic)', 'Electronic check': 'Electronic check'}
    }

    for col, mapping in cat_mappings.items():
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].map(mapping).fillna('Unknown')

    return df

# Batch prediction
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        cleaned_data = clean_and_validate_data(data)
        filtered_data = cleaned_data[cleaned_data['Tenure'] >= min_tenure].copy()

        if filtered_data.empty:
            st.warning(f"⚠️ Tidak ada data dengan Tenure ≥ {min_tenure}")
            st.stop()

        st.write(f"📋 {filtered_data.shape[0]} data dengan Tenure ≥ {min_tenure} ditemukan.")
        st.dataframe(filtered_data.head())

        if st.button("🔮 Prediksi Churn Batch", key="batch_predict", help="Klik untuk memprediksi churn pelanggan secara batch"):
            if model is not None:
                try:
                    pred = model.predict(filtered_data)
                    proba = model.predict_proba(filtered_data)[:, 1]

                    results = filtered_data.copy()
                    results['Churn_Predicted'] = pred
                    results['Proba_Churn'] = (proba * 100).round(2)
                    results['Churn_Risk'] = pd.cut(results['Proba_Churn'],
                                                  bins=[0, 30, 70, 100],
                                                  labels=['Low', 'Medium', 'High'])

                    st.success("✅ Prediksi batch berhasil dilakukan!")

                    display_cols = ['Churn_Predicted', 'Proba_Churn', 'Churn_Risk'] + \
                                   [col for col in results.columns if col not in ['Churn_Predicted', 'Proba_Churn', 'Churn_Risk']]

                    st.dataframe(results[display_cols])

                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Hasil Prediksi Batch",
                        data=csv,
                        file_name="hasil_prediksi_churn_batch.csv",
                        mime="text/csv"
                    )

                except NotFittedError:
                    st.error("❌ Model belum di-fit. Pastikan model yang dimuat sudah terlatih.")
                except Exception as e:
                    st.error(f"❌ Terjadi error saat prediksi batch: {str(e)}")
            else:
                st.warning("⚠️ Model belum dimuat. Silakan coba lagi atau hubungi administrator.")
    except Exception as e:
        st.error(f"❌ Error saat memproses data batch: {str(e)}")

# Manual input form
st.sidebar.markdown("---")
st.sidebar.subheader("Input Manual Pelanggan")

def user_input_features():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("SeniorCitizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
    tenure = st.sidebar.number_input("Tenure (bulan)", min_value=0, max_value=100, value=12)
    phone_service = st.sidebar.selectbox("PhoneService", ["No", "Yes"])
    paperless_billing = st.sidebar.selectbox("PaperlessBilling", ["No", "Yes"])
    multiple_lines = st.sidebar.selectbox("MultipleLines", ["No", "Yes", "No phone service"])
    internet_service = st.sidebar.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("OnlineBackup", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("DeviceProtection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("TechSupport", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("StreamingTV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("StreamingMovies", ["No", "Yes", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.sidebar.selectbox("PaymentMethod", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.sidebar.number_input("MonthlyCharges", min_value=0.0, value=70.0, step=0.1)
    total_charges = st.sidebar.number_input("TotalCharges", min_value=0.0, value=1000.0, step=0.1)

    data = {
        'Gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'Tenure': tenure,
        'PhoneService': phone_service,
        'PaperlessBilling': paperless_billing,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    features = pd.DataFrame([data])
    return features

manual_df = user_input_features()

if st.sidebar.button("🔮 Prediksi Churn Manual"):
    if model is not None:
        try:
            manual_cleaned = clean_and_validate_data(manual_df)
            pred = model.predict(manual_cleaned)
            proba = model.predict_proba(manual_cleaned)[:, 1]

            manual_cleaned['Churn_Predicted'] = pred
            manual_cleaned['Proba_Churn'] = (proba * 100).round(2)
            manual_cleaned['Churn_Risk'] = pd.cut(manual_cleaned['Proba_Churn'],
                                                 bins=[0, 30, 70, 100],
                                                 labels=['Low', 'Medium', 'High'])

            st.success("✅ Prediksi manual berhasil!")
            display_cols = ['Churn_Predicted', 'Proba_Churn', 'Churn_Risk'] + \
                           [col for col in manual_cleaned.columns if col not in ['Churn_Predicted', 'Proba_Churn', 'Churn_Risk']]

            st.dataframe(manual_cleaned[display_cols])

        except Exception as e:
            st.error(f"❌ Terjadi error saat prediksi manual: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
    else:
        st.warning("⚠️ Model belum dimuat. Silakan coba lagi atau hubungi administrator.")
