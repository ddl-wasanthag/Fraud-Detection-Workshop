import streamlit as st
import time
import random
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    .fraud-alert {
        background-color: #ff4b4b;
        color: white;
        border: 2px solid #ff0000;
    }
    .safe-alert {
        background-color: #00cc88;
        color: white;
        border: 2px solid #00aa66;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 5px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🔒 Fraud Detection System</h1>', unsafe_allow_html=True)

# Create three columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("💰 Transaction Details")
    amount = st.number_input("Amount", min_value=0.01, value=50.0, step=0.01)
    hour = st.selectbox("Hour", range(24), index=12)
    tx_type = st.selectbox("Transaction Type", ["purchase", "transfer", "payment", "withdrawal"])
    card_present = st.selectbox("Card Present", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col2:
    st.subheader("👤 Customer Profile")
    age = st.slider("Age", 18, 80, 35)
    tenure = st.slider("Tenure (years)", 0, 30, 5)
    txn_24h = st.number_input("Transactions in 24h", min_value=0, value=1, step=1)
    avg_30d = st.number_input("Avg 30d Amount", value=0.0, step=0.01)

with col3:
    st.subheader("🔍 Risk Factors")
    merchant_risk = st.slider("Merchant Risk", -3.0, 3.0, 0.0, 0.1)
    device_trust = st.slider("Device Trust", -3.0, 3.0, 0.0, 0.1)
    ip_reputation = st.slider("IP Reputation", -3.0, 3.0, 0.0, 0.1)
    dist_from_home = st.slider("Distance from Home", -2.0, 5.0, 0.0, 0.1)

# Location and merchant details
st.subheader("📍 Location & Merchant")
col4, col5 = st.columns(2)

with col4:
    latitude = st.number_input("Latitude", value=40.0, step=0.01)
    longitude = st.number_input("Longitude", value=-75.0, step=0.01)
    
with col5:
    device_type = st.selectbox("Device Type", ["mobile", "desktop", "ATM", "tablet"])
    merchant_cat = st.selectbox("Merchant Category", 
                               ["grocery", "gas", "electronics", "travel", "clothing", "entertainment", "restaurant"])
    channel = st.selectbox("Channel", ["online", "in-store", "contactless", "chip"])

# Prediction button and results
st.markdown("---")
predict_button = st.button("🔍 Predict Fraud Risk", type="primary")

if predict_button:
    # Show loading spinner
    with st.spinner("Analyzing transaction... 🔍"):
        time.sleep(2)  # Simulate API call
        
        # Dummy prediction logic (you can replace this with actual model)
        risk_score = random.uniform(0, 1)
        
        # Simple heuristic for demo
        risk_factors = [
            amount > 500,
            merchant_risk > 1.0,
            device_trust < -1.0,
            ip_reputation < -1.0,
            dist_from_home > 2.0,
            hour in [0, 1, 2, 3, 4, 5, 23]
        ]
        
        risk_score = sum(risk_factors) / len(risk_factors)
        is_fraud = risk_score > 0.4
        
        # Display results
        if is_fraud:
            st.markdown(f"""
            <div class="prediction-box fraud-alert">
                <h2>⚠️ FRAUD ALERT</h2>
                <h3>Risk Score: {risk_score:.2%}</h3>
                <p>This transaction has been flagged as potentially fraudulent.</p>
                <p>Please review manually before processing.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box safe-alert">
                <h2>✅ TRANSACTION APPROVED</h2>
                <h3>Risk Score: {risk_score:.2%}</h3>
                <p>This transaction appears to be legitimate.</p>
                <p>Safe to process.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show risk factors breakdown
        st.subheader("📊 Risk Analysis")
        
        risk_data = {
            "Factor": ["High Amount", "Merchant Risk", "Device Trust", "IP Reputation", "Distance", "Unusual Hour"],
            "Score": [amount/1000, merchant_risk, device_trust, ip_reputation, dist_from_home, 1 if hour in [0,1,2,3,4,5,23] else 0],
            "Status": ["⚠️" if factor else "✅" for factor in risk_factors]
        }
        
        df = pd.DataFrame(risk_data)
        st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>🔒 Fraud Detection System v1.0 | Built with Streamlit</p>",
    unsafe_allow_html=True
)