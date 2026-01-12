import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Client Churn Prediction", layout="wide")

# Title
st.title("🏦 Client Churn Prediction & Analytics")

# Load data from CSV
@st.cache_data
def load_data():
    # Try to load from the specified path
    csv_path = r"C:\Users\manishm\Desktop\Ml.py\sample_customer_transactions_2000_rows_with_inactivity_logic.csv"
    
    try:
        df = pd.read_csv(csv_path)
        st.sidebar.success(f"✅ Loaded {len(df)} rows from CSV")
    except FileNotFoundError:
        st.sidebar.error("❌ CSV file not found. Using sample data.")
        # Fallback to sample data
        data = {
            'Client_ID': ['C011', 'C012'],
            'Client_Name': ['Rakesh Shah', 'Meenal Deshpande'],
            'Account_No': ['ACC5012', 'ACC7912'],
            'Transaction_Date': ['19/12/2025', '20/12/2025'],
            'Transaction_Type': ['Gross Sales', 'Redemption'],
            'Product': ['Equity', 'Mutual Fund'],
            'Amount': [586778, -192323],
            'Currency': ['INR', 'INR'],
            'Channel': ['Branch', 'Online'],
            'City': ['Surat', 'Nashik'],
            'Days_Since_Last_Transaction': [10, 11]
        }
        df = pd.DataFrame(data)
    except Exception as e:
        st.sidebar.error(f"❌ Error loading CSV: {str(e)}")
        st.stop()
    
    # Clean column names (remove spaces, standardize)
    df.columns = df.columns.str.strip()
    
    # Feature engineering - check which columns exist
    if 'Amount' in df.columns:
        df['Amount_Abs'] = df['Amount'].abs()
        df['Is_Negative'] = (df['Amount'] < 0).astype(int)
    else:
        df['Amount_Abs'] = 0
        df['Is_Negative'] = 0
    
    # Add transaction frequency if not present
    if 'Transaction_Frequency' not in df.columns:
        # Group by client and count transactions
        if 'Client_ID' in df.columns:
            freq = df.groupby('Client_ID').size().reset_index(name='Transaction_Frequency')
            df = df.merge(freq, on='Client_ID', how='left')
        else:
            df['Transaction_Frequency'] = np.random.randint(1, 20, size=len(df))
    
    # Add account age if not present
    if 'Account_Age_Days' not in df.columns:
        df['Account_Age_Days'] = np.random.randint(30, 1825, size=len(df))
    
    # Calculate average transaction amount
    df['Avg_Transaction_Amount'] = df['Amount_Abs'] / (df['Transaction_Frequency'] + 1)
    
    # Ensure Days_Since_Last_Transaction exists
    if 'Days_Since_Last_Transaction' not in df.columns:
        df['Days_Since_Last_Transaction'] = np.random.randint(1, 90, size=len(df))
    
    # Create churn label based on business logic
    # Higher churn probability for: long inactivity, negative transactions, low frequency
    churn_score = (
        (df['Days_Since_Last_Transaction'] / 90) * 0.4 +
        (df['Is_Negative']) * 0.3 +
        (1 - df['Transaction_Frequency'] / df['Transaction_Frequency'].max()) * 0.3
    )
    df['Churn'] = (churn_score + np.random.normal(0, 0.1, len(df)) > 0.5).astype(int)
    
    return df

df = load_data()

# Sidebar
st.sidebar.header("📊 Dashboard Controls")
show_data = st.sidebar.checkbox("Show Raw Data", value=False)

# Check if City column exists for filtering
if 'City' in df.columns:
    city_filter = st.sidebar.multiselect("Filter by City", options=df['City'].unique(), default=df['City'].unique())
    df_filtered = df[df['City'].isin(city_filter)]
else:
    st.sidebar.info("City column not found in data")
    df_filtered = df

# Display raw data if requested
if show_data:
    st.subheader("📋 Raw Data")
    st.dataframe(df_filtered, use_container_width=True)

# Train ML Model
@st.cache_resource
def train_model(data):
    # Prepare features
    le_channel = LabelEncoder()
    le_product = LabelEncoder()
    le_city = LabelEncoder()
    le_trans_type = LabelEncoder()
    
    # Base numerical features
    X = data[['Days_Since_Last_Transaction', 'Amount_Abs', 'Is_Negative', 
              'Transaction_Frequency', 'Account_Age_Days', 'Avg_Transaction_Amount']].copy()
    
    # Encode categorical features if they exist
    if 'Channel' in data.columns:
        X['Channel_Encoded'] = le_channel.fit_transform(data['Channel'].fillna('Unknown'))
    else:
        X['Channel_Encoded'] = 0
        
    if 'Product' in data.columns:
        X['Product_Encoded'] = le_product.fit_transform(data['Product'].fillna('Unknown'))
    else:
        X['Product_Encoded'] = 0
        
    if 'City' in data.columns:
        X['City_Encoded'] = le_city.fit_transform(data['City'].fillna('Unknown'))
    else:
        X['City_Encoded'] = 0
        
    if 'Transaction_Type' in data.columns:
        X['TransType_Encoded'] = le_trans_type.fit_transform(data['Transaction_Type'].fillna('Unknown'))
    else:
        X['TransType_Encoded'] = 0
    
    y = data['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Get predictions
    data['Churn_Probability'] = model.predict_proba(X)[:, 1]
    data['Predicted_Churn'] = model.predict(X)
    
    accuracy = model.score(X_test, y_test)
    
    return model, accuracy, data, le_channel, le_product, le_city, le_trans_type

model, accuracy, df_predicted, le_channel, le_product, le_city, le_trans_type = train_model(df)

# Update filtered data with predictions
if 'City' in df_predicted.columns:
    df_filtered = df_predicted[df_predicted['City'].isin(city_filter)]
else:
    df_filtered = df_predicted

# Key Metrics
st.subheader("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Clients", len(df_filtered))
with col2:
    likely_to_leave = (df_filtered['Predicted_Churn'] == 1).sum()
    st.metric("Likely to Leave", likely_to_leave, delta=f"{likely_to_leave/len(df_filtered)*100:.1f}%", delta_color="inverse")
with col3:
    likely_to_stay = (df_filtered['Predicted_Churn'] == 0).sum()
    st.metric("Likely to Stay", likely_to_stay, delta=f"{likely_to_stay/len(df_filtered)*100:.1f}%")
with col4:
    st.metric("Model Accuracy", f"{accuracy*100:.1f}%")

# Create two columns for charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Churn Prediction Distribution")
    churn_counts = df_filtered['Predicted_Churn'].value_counts()
    fig_pie = px.pie(
        values=churn_counts.values,
        names=['Will Stay', 'Will Leave'],
        color_discrete_sequence=['#ADD8E6', '#F0F2F6'],
        hole=0.4
    )
    fig_pie.update_traces(
        textposition='inside', 
        textinfo='percent+label+value',
        textfont_size=14
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("📍 Clients by City")
    if 'City' in df_filtered.columns:
        city_counts = df_filtered['City'].value_counts().reset_index()
        city_counts.columns = ['City', 'Count']
        fig_bar = px.bar(
            city_counts,
            x='City',
            y='Count',
            color='Count',
            color_continuous_scale='Blues',
            text='Count'
        )
        fig_bar.update_traces(textposition='outside', textfont_size=12)
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("City data not available")

# City-wise Churn Analysis
if 'City' in df_filtered.columns:
    st.subheader("🗺️ City-wise Churn Prediction")
    city_churn = df_filtered.groupby(['City', 'Predicted_Churn']).size().reset_index(name='Count')
    city_churn['Status'] = city_churn['Predicted_Churn'].map({0: 'Will Stay', 1: 'Will Leave'})

    fig_city = px.bar(
        city_churn,
        x='City',
        y='Count',
        color='Status',
        barmode='group',
        color_discrete_map={'Will Stay': '#00CC96', 'Will Leave': '#EF553B'},
        text='Count'
    )
    fig_city.update_traces(textposition='outside', textfont_size=11)
    st.plotly_chart(fig_city, use_container_width=True)

# Risk Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("⚠️ High Risk Clients")
    high_risk_cols = ['Client_Name', 'City', 'Days_Since_Last_Transaction', 'Churn_Probability'] if 'Client_Name' in df_filtered.columns and 'City' in df_filtered.columns else ['Days_Since_Last_Transaction', 'Churn_Probability']
    if 'Client_ID' in df_filtered.columns and 'Client_Name' not in df_filtered.columns:
        high_risk_cols = ['Client_ID', 'City', 'Days_Since_Last_Transaction', 'Churn_Probability'] if 'City' in df_filtered.columns else ['Client_ID', 'Days_Since_Last_Transaction', 'Churn_Probability']
    
    high_risk = df_filtered[df_filtered['Churn_Probability'] > 0.7].sort_values('Churn_Probability', ascending=False)[high_risk_cols].head(10)
    if not high_risk.empty:
        high_risk['Churn_Probability'] = (high_risk['Churn_Probability'] * 100).round(1).astype(str) + '%'
        st.dataframe(high_risk, use_container_width=True, hide_index=True)
    else:
        st.info("No high-risk clients identified")

with col2:
    st.subheader("✅ Loyal Clients")
    loyal_cols = ['Client_Name', 'City', 'Transaction_Frequency', 'Churn_Probability'] if 'Client_Name' in df_filtered.columns and 'City' in df_filtered.columns else ['Transaction_Frequency', 'Churn_Probability']
    if 'Client_ID' in df_filtered.columns and 'Client_Name' not in df_filtered.columns:
        loyal_cols = ['Client_ID', 'City', 'Transaction_Frequency', 'Churn_Probability'] if 'City' in df_filtered.columns else ['Client_ID', 'Transaction_Frequency', 'Churn_Probability']
    
    loyal = df_filtered[df_filtered['Churn_Probability'] < 0.3].sort_values('Churn_Probability')[loyal_cols].head(10)
    if not loyal.empty:
        loyal['Churn_Probability'] = (loyal['Churn_Probability'] * 100).round(1).astype(str) + '%'
        st.dataframe(loyal, use_container_width=True, hide_index=True)
    else:
        st.info("No loyal clients identified")

# Feature Importance
st.subheader("🔍 Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': ['Days Since Last Txn', 'Transaction Amount', 'Negative Txn', 
                'Transaction Frequency', 'Account Age', 'Avg Txn Amount',
                'Channel', 'Product', 'City', 'Transaction Type'],
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig_importance = px.bar(
    feature_importance,
    x='Importance',
    y='Feature',
    orientation='h',
    color='Importance',
    color_continuous_scale='Viridis',
    text='Importance'
)
fig_importance.update_traces(
    texttemplate='%{text:.3f}',
    textposition='outside',
    textfont_size=11
)
fig_importance.update_layout(showlegend=False)
st.plotly_chart(fig_importance, use_container_width=True)

# Future Predictions Summary
st.subheader("🔮 Future Outlook Summary")
col1, col2, col3 = st.columns(3)

with col1:
    avg_churn_prob = df_filtered['Churn_Probability'].mean()
    st.metric("Avg Churn Risk", f"{avg_churn_prob*100:.1f}%")

with col2:
    high_risk_pct = (df_filtered['Churn_Probability'] > 0.7).sum() / len(df_filtered) * 100
    st.metric("High Risk Clients %", f"{high_risk_pct:.1f}%")

with col3:
    avg_inactivity = df_filtered['Days_Since_Last_Transaction'].mean()
    st.metric("Avg Days Inactive", f"{avg_inactivity:.0f}")

# Recommendations
st.subheader("💡 Recommendations")
st.info("""
**Action Items:**
- 🎯 Focus on clients with >70% churn probability for immediate retention campaigns
- 📞 Contact clients inactive for >30 days
- 🎁 Offer personalized incentives to high-value clients showing churn signals
- 📊 Monitor transaction patterns in cities with higher churn rates
- 💳 Encourage digital channel adoption for better engagement
""")

# Footer
st.markdown("---")
st.caption("🤖 ML Model: Random Forest Classifier | Data updated in real-time")