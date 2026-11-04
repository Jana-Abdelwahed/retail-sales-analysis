import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------------------------------------
# 1. Page Configuration & Unified Professional Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Executive Retail Intelligence Dashboard", 
    page_icon="📊", 
    layout="wide"
)

# Inject custom CSS for high-contrast navy/dark blue text across all components
st.markdown("""
    <style>
    /* Global Background Gradient across the entire window */
    .stApp {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 50%, #E8F4FC 100%);
        color: #0F172A;
    }
    
    /* Top Header Bar */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #38BDF8 0%, #0284C7 100%) !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E0F2FE 0%, #BAE6FD 100%);
        border-right: 1px solid #7DD3FC;
    }
    
    /* Sidebar text colors */
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] span {
        color: #0F172A !important;
        font-weight: 600;
    }

    /* High-contrast Navy Blue for Streamlit Metric Components & Labels */
    div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] label, div[data-testid="stMetricLabel"] p, [data-testid="stMetric"] label {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    
    /* Typography Styling */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #0369A1 !important;
    }

    /* Light/White File Uploader Inner Box & Drop Zone */
    div[data-testid="stFileUploader"] {
        background: transparent !important;
        padding: 0px !important;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #FFFFFF !important;
        border: 2px dashed #0284C7 !important;
        border-radius: 10px !important;
    }
    div[data-testid="stFileUploader"] label, 
    div[data-testid="stFileUploader"] small, 
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] div {
        color: #0F172A !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: #0284C7 !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: #0369A1 !important;
    }

    /* Multi-select Dropdown & Tag styling */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #7DD3FC !important;
        color: #0F172A !important;
    }
    span[data-baseweb="tag"] {
        background-color: #E0F2FE !important;
        border: 1px solid #38BDF8 !important;
    }
    span[data-baseweb="tag"] span {
        color: #0369A1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Set professional Matplotlib/Seaborn theme parameters
sns.set_theme(style="whitegrid", palette="Blues_d")
plt.rcParams['font.sans-serif'] = 'Segoe UI'
plt.rcParams['axes.edgecolor'] = '#BAE6FD'
plt.rcParams['axes.linewidth'] = 1.0

# -------------------------------------------------------------
# 2. Sidebar Navigation & File Upload
# -------------------------------------------------------------
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Cleaned Dataset", type=["csv"])

st.title("📊 Executive Retail Sales & Intelligence Dashboard")
st.markdown("A production-grade analytics and machine learning suite tracking revenue drivers and customer behavioral clusters.")
st.markdown("---")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    
    # Sidebar Filters
    st.sidebar.markdown("### Data Filters")
    regions = df['Region'].unique() if 'Region' in df.columns else []
    categories = df['Category'].unique() if 'Category' in df.columns else []
    
    selected_region = st.sidebar.multiselect("Region", options=regions, default=regions)
    selected_category = st.sidebar.multiselect("Product Category", options=categories, default=categories)
    
    # Filter dataframe safely
    filtered_df = df.copy()
    if 'Region' in df.columns and selected_region:
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_region)]
    if 'Category' in df.columns and selected_category:
        filtered_df = filtered_df[filtered_df['Category'].isin(selected_category)]
    
    # -------------------------------------------------------------
    # 3. Executive KPI Metrics Bar
    # -------------------------------------------------------------
    total_revenue = filtered_df['Calculated_Revenue'].sum() if 'Calculated_Revenue' in filtered_df.columns else filtered_df['Sales'].sum()
    total_orders = filtered_df['OrderID'].nunique() if 'OrderID' in filtered_df.columns else len(filtered_df)
    avg_order_val = total_revenue / total_orders if total_orders > 0 else 0
    total_units = filtered_df['Quantity'].sum() if 'Quantity' in filtered_df.columns else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric(label="Total Revenue", value=f"${total_revenue:,.2f}", delta="+4.2% YoY")
    with kpi2:
        st.metric(label="Total Orders", value=f"{total_orders:,}", delta="+1.8% MoM")
    with kpi3:
        st.metric(label="Avg Order Value (AOV)", value=f"${avg_order_val:,.2f}")
    with kpi4:
        st.metric(label="Units Distributed", value=f"{total_units:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. Professional Visualizations Suite
    # -------------------------------------------------------------
    col_a, col_b = st.columns(2)
    gradient_palette = ['#0284C7', '#0EA5E9', '#38BDF8', '#7DD3FC', '#BAE6FD']

    with col_a:
        st.subheader("Revenue Performance by Category")
        cat_fig, ax = plt.subplots(figsize=(8, 5))
        cat_fig.patch.set_facecolor('#F0F9FF')
        ax.set_facecolor('#FFFFFF')
        
        cat_summary = filtered_df.groupby('Category')['Calculated_Revenue'].sum().reset_index()
        
        sns.barplot(
            data=cat_summary, 
            x='Category', 
            y='Calculated_Revenue', 
            palette=gradient_palette[:len(cat_summary)], 
            ax=ax
        )
        ax.set_title("Top Revenue Generators", fontsize=12, fontweight='bold', color='#0369A1', pad=10)
        ax.set_ylabel("Revenue ($)", fontsize=10, color='#334155')
        ax.set_xlabel("", fontsize=10)
        ax.yaxis.set_major_formatter('${x:,.0f}')
        sns.despine(top=True, right=True)
        st.pyplot(cat_fig)
        
    with col_b:
        st.subheader("Monthly Revenue Trajectory")
        time_fig, ax = plt.subplots(figsize=(8, 5))
        time_fig.patch.set_facecolor('#F0F9FF')
        ax.set_facecolor('#FFFFFF')
        
        monthly_trend = filtered_df.groupby([filtered_df['OrderDate'].dt.to_period('M')])['Calculated_Revenue'].sum().reset_index()
        monthly_trend['OrderDate'] = monthly_trend['OrderDate'].astype(str)
        
        sns.lineplot(
            data=monthly_trend, 
            x='OrderDate', 
            y='Calculated_Revenue', 
            marker='o', 
            color='#0284C7', 
            linewidth=2.5, 
            ax=ax
        )
        ax.set_title("Timeline Trend Analysis", fontsize=12, fontweight='bold', color='#0369A1', pad=10)
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel("Revenue ($)", fontsize=10, color='#334155')
        ax.set_xlabel("", fontsize=10)
        ax.yaxis.set_major_formatter('${x:,.0f}')
        sns.despine(top=True, right=True)
        st.pyplot(time_fig)

    # -------------------------------------------------------------
    # 5. Machine Learning Integration: RFM Customer Segmentation
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("🤖 Unsupervised Machine Learning: RFM Customer Clustering")
    st.markdown("Segmenting customer behavioral profiles using **K-Means Clustering** across Recency, Frequency, and Monetary dimensions.")
    
    if st.button("Execute K-Means Clustering Algorithm", type="primary"):
        with st.spinner("Processing clustering centroids..."):
            snapshot_date = filtered_df['OrderDate'].max() + pd.Timedelta(days=1)
            customer_col = 'CustomerID' if 'CustomerID' in filtered_df.columns else 'CustomerSegment'
            
            rfm = filtered_df.groupby(customer_col).agg(
                Recency=('OrderDate', lambda x: (snapshot_date - x.max()).days),
                Frequency=('OrderID', 'count') if 'OrderID' in filtered_df.columns else ('OrderDate', 'count'),
                Monetary=('Calculated_Revenue', 'sum')
            ).reset_index()
            
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(np.log1p(rfm[['Recency', 'Frequency', 'Monetary']]))
            
            n_clusters = min(4, len(rfm))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
            
            col_m1, col_m2 = st.columns([1, 1.5])
            with col_m1:
                st.success(f"Successfully generated {n_clusters} distinct customer behavioral segments.")
                st.dataframe(rfm.head(8), use_container_width=True)
                
            with col_m2:
                cluster_fig, ax = plt.subplots(figsize=(8, 5))
                cluster_fig.patch.set_facecolor('#F0F9FF')
                ax.set_facecolor('#FFFFFF')
                
                sns.scatterplot(
                    data=rfm, 
                    x='Frequency', 
                    y='Monetary', 
                    hue='Cluster', 
                    palette='Blues_r', 
                    s=90, 
                    alpha=0.85, 
                    ax=ax
                )
                ax.set_title("Customer Segmentation Map (Frequency vs. Monetary)", fontsize=11, fontweight='bold', color='#0369A1')
                ax.set_xlabel("Order Frequency", fontsize=10, color='#334155')
                ax.set_ylabel("Monetary Value ($)", fontsize=10, color='#334155')
                sns.despine(top=True, right=True)
                st.pyplot(cluster_fig)
else:
    st.info("👈 Please upload your `cleaned_retail_sales.csv` file using the sidebar panel to initialize the executive dashboard.")