# Home.py
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #F0F8FF;
    }
    .main-header {
        font-size: 3rem;
        color: #007ACC;
        text-align: center;
        margin: 2rem 0;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .description {
        font-size: 1.2rem;
        line-height: 1.6;
        color: #FFFFFF;
        text-align: justify;
        margin: 1rem auto;
        width: 80%;
    }
    .feature-box {
        background-color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem auto;
        width: 80%;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .cta-button {
        display: block;
        width: fit-content;
        margin: 2rem auto;
        padding: 1rem 2rem;
        background-color: #007ACC;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        border-radius: 25px;
        text-decoration: none;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease;
    }
    .cta-button:hover {
        background-color: #005F99;
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.markdown('<h1 class="main-header">📱 Welcome to the Telco Customer Churn Prediction 📱</h1>', unsafe_allow_html=True)

# Sub Header
st.markdown('<h2 class="sub-header">Predict Customer Churn and Improve Retention Strategies</h2>', unsafe_allow_html=True)

# Description
st.markdown("""
    <div class="description">
        This application uses advanced machine learning techniques to predict the likelihood of customer churn 
        in your telecom company. By analyzing customer behavior, contract details, and service usage patterns, 
        you can proactively identify at-risk customers and take actions to retain them.
    </div>
""", unsafe_allow_html=True)

# How It Works section with better contrast
st.markdown("""
    <div class="feature-box" style="background-color: #FFFFFF; color: black; border: 1px solid #B0BEC5;">
        <h3>🔍 How It Works:</h3>
        <ol style="line-height: 1.8;">
            <li><strong>Upload Customer Data:</strong> Upload your CSV file containing customer information and usage details.</li>
            <li><strong>Filter Data:</strong> Apply filters such as minimum tenure to focus on relevant customers.</li>
            <li><strong>Run Predictions:</strong> Use our trained machine learning model to predict which customers are likely to churn.</li>
            <li><strong>Analyze Results:</strong> View prediction results with probabilities and export them for further analysis.</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <footer style="text-align:center; margin-top:3rem; font-size:0.9rem;">
        Telco Customer Churn Prediction App - Powered by Machine Learning V1. Contact your system administrator for assistance.
    </footer>
""", unsafe_allow_html=True)
