import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #2e3440;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .prediction-value {
        font-size: 3rem;
        color: #00ff00;
        font-weight: bold;
    }
    .feature-input {
        background-color: #3b4252;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "http://localhost:8000/api/v1"

def main():
    st.markdown('<h1 class="main-header">🏠 House Price Prediction System</h1>', unsafe_allow_html=True)
    
    # Sidebar for input
    st.sidebar.title("House Features")
    st.sidebar.markdown("Enter the property details below:")
    
    # Input fields
    sqft = st.sidebar.number_input(
        "Square Footage (SQFT)", 
        min_value=100, 
        max_value=5000, 
        value=1500,
        step=100
    )
    
    bedrooms = st.sidebar.number_input(
        "Number of Bedrooms", 
        min_value=1, 
        max_value=10, 
        value=3,
        step=1
    )
    
    location = st.sidebar.selectbox(
        "Location Type",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Urban", 2: "Suburban", 3: "Rural"}[x]
    )
    
    region = st.sidebar.selectbox(
        "Region",
        options=[1, 2, 3, 4],
        format_func=lambda x: {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}[x]
    )
    
    titled = st.sidebar.selectbox(
        "Title Type",
        options=[1, 2, 3],
        format_func=lambda x: {1: "Vehicle", 2: "Land-Home", 3: "Other"}[x]
    )
    
    lease = st.sidebar.selectbox(
        "Lease",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    
    footings = st.sidebar.number_input(
        "Footing Type",
        min_value=1,
        max_value=9,
        value=1,
        step=1
    )
    
    # Create feature dict
    features = {
        "SQFT": sqft,
        "BEDROOMS": bedrooms,
        "LOCATION": location,
        "REGION": region,
        "TITLED": titled,
        "LEASE": lease,
        "FOOTINGS": footings
    }
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Feature Summary")
        
        # Display features in a nice table
        feature_df = pd.DataFrame({
            "Feature": ["Square Footage", "Bedrooms", "Location", "Region", "Title", "Lease", "Footings"],
            "Value": [
                f"{sqft} sqft",
                bedrooms,
                {1: "Urban", 2: "Suburban", 3: "Rural"}[location],
                {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}[region],
                {1: "Vehicle", 2: "Land-Home", 3: "Other"}[titled],
                "Yes" if lease == 1 else "No",
                footings
            ]
        })
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
        
        # Prediction button
        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            with st.spinner("Making prediction..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"features": features}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        predicted_price = result["predicted_price"]
                        
                        # Display prediction
                        st.markdown(f"""
                            <div class="prediction-box">
                                <h2>Estimated House Price</h2>
                                <div class="prediction-value">${predicted_price:,.2f}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional info
                        st.info(f"📍 Prediction made at {result['timestamp']}")
                        
                        # Add to history
                        if "history" not in st.session_state:
                            st.session_state.history = []
                        
                        st.session_state.history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "price": predicted_price,
                            "features": features
                        })
                        
                    else:
                        st.error(f"Error: {response.text}")
                
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")
                    st.info("Make sure the API server is running: `make run-api`")
    
    with col2:
        st.subheader("📈 Statistics")
        
        # Show model info
        try:
            response = requests.get(f"{API_URL}/model/info")
            if response.status_code == 200:
                model_info = response.json()
                st.metric("🤖 Model", model_info["model_name"])
                
                if "metrics" in model_info:
                    metrics = model_info["metrics"]
                    st.metric("📊 R² Score", f"{metrics.get('r2_score', 0):.4f}")
                    st.metric("📉 RMSE", f"${metrics.get('rmse', 0):,.2f}")
        except:
            st.warning("Cannot connect to model info endpoint")
        
        # Price distribution if there's history
        if "history" in st.session_state and len(st.session_state.history) > 0:
            st.subheader("📊 Prediction History")
            history_df = pd.DataFrame(st.session_state.history)
            fig = px.line(
                history_df,
                x="timestamp",
                y="price",
                title="Price Predictions Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Bottom section - Batch prediction
    st.markdown("---")
    st.subheader("📋 Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file for batch prediction",
        type=["csv"],
        help="Upload a CSV file with the same features as above"
    )
    
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(batch_df.head())
            
            if st.button("Run Batch Prediction"):
                with st.spinner("Processing batch predictions..."):
                    # Convert to list of dicts
                    batch_features = batch_df.to_dict(orient="records")
                    
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        json={"features": batch_features}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Predictions", result["total_predictions"])
                        with col2:
                            st.metric("Average Price", f"${result['avg_price']:,.2f}")
                        with col3:
                            st.metric("Price Range", f"${result['min_price']:,.2f} - ${result['max_price']:,.2f}")
                        
                        # Show detailed results
                        predictions_df = pd.DataFrame([
                            {
                                "Predicted Price": p["predicted_price"],
                                "Formatted": p["predicted_price_formatted"],
                                **p["input_features"]
                            }
                            for p in result["predictions"]
                        ])
                        st.dataframe(predictions_df, use_container_width=True)
                        
                        # Plot distribution
                        fig = px.histogram(
                            predictions_df,
                            x="Predicted Price",
                            title="Distribution of Predicted Prices",
                            nbins=30
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()