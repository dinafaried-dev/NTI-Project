import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, recall_score
from imblearn.over_sampling import SMOTE
import io

# Page configuration
st.set_page_config(
    page_title="Customer Churn Analysis & Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert > div {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Customer Churn Analysis & Prediction</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "🏠 Home & Data Overview",
    "📊 Data Exploration", 
    "📈 Data Visualization",
    "🤖 Machine Learning Models",
    "🔧 Model Optimization",
    "📄 Upload Your Data"
])

# Load data function
@st.cache_data
def load_sample_data():
    """Create sample data similar to the credit card churn dataset"""
    np.random.seed(42)
    n_samples = 10000
    
    # Generate customer data
    data = {
        'Customer_Age': np.random.randint(18, 80, n_samples),
        'Gender': np.random.choice(['M', 'F'], n_samples),
        'Dependent_count': np.random.randint(0, 6, n_samples),
        'Education_Level': np.random.choice([
            'High School', 'Graduate', 'College', 'Unknown', 
            'Uneducated', 'Post-Graduate'
        ], n_samples),
        'Marital_Status': np.random.choice([
            'Married', 'Single', 'Divorced', 'Unknown'
        ], n_samples),
        'Income_Category': np.random.choice([
            'Less than $40K', '$40K - $60K', '$60K - $80K', 
            '$80K - $120K', '$120K +'
        ], n_samples),
        'Card_Category': np.random.choice([
            'Blue', 'Silver', 'Gold', 'Platinum'
        ], n_samples, p=[0.7, 0.15, 0.1, 0.05]),
        'Months_on_book': np.random.randint(13, 56, n_samples),
        'Total_Relationship_Count': np.random.randint(1, 7, n_samples),
        'Months_Inactive_12_mon': np.random.randint(0, 7, n_samples),
        'Contacts_Count_12_mon': np.random.randint(0, 7, n_samples),
        'Credit_Limit': np.random.randint(1000, 35000, n_samples),
        'Total_Revolving_Bal': np.random.randint(0, 3000, n_samples),
        'Total_Trans_Amt': np.random.randint(500, 5000, n_samples),
        'Total_Trans_Ct': np.random.randint(10, 140, n_samples),
        'Avg_Utilization_Ratio': np.random.uniform(0, 1, n_samples),
        'Total_Amt_Chng_Q4_Q1': np.random.uniform(0.5, 3.0, n_samples),
        'Total_Ct_Chng_Q4_Q1': np.random.uniform(0.5, 3.0, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate Avg_Open_To_Buy
    df['Avg_Open_To_Buy'] = df['Credit_Limit'] - df['Total_Revolving_Bal']
    
    # Create realistic churn based on multiple factors
    churn_probability = (
        (df['Months_Inactive_12_mon'] / 6) * 0.25 +
        (df['Contacts_Count_12_mon'] / 6) * 0.15 +
        (1 - df['Total_Trans_Ct'] / 140) * 0.25 +
        (df['Avg_Utilization_Ratio']) * 0.15 +
        np.where(df['Total_Relationship_Count'] <= 2, 0.2, 0)
    )
    
    df['Attrition_Flag'] = np.where(
        np.random.random(n_samples) < churn_probability,
        'Attrited Customer',
        'Existing Customer'
    )
    
    return df

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = load_sample_data()

# File upload function
def upload_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Remove the Naive Bayes columns if they exist
            cols_to_remove = [col for col in df.columns if 'Naive_Bayes_Classifier' in col]
            if cols_to_remove:
                df = df.drop(columns=cols_to_remove)
            if 'CLIENTNUM' in df.columns:
                df = df.drop(columns=['CLIENTNUM'])
            st.success("✅ File uploaded successfully!")
            return df
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return None
    return None

# Value counts function
def display_value_counts(data, column):
    """Display value counts with percentages"""
    value_counts = data[column].value_counts(normalize=True, dropna=False) * 100
    formatted_data = pd.DataFrame({
        'Count': data[column].value_counts(dropna=False),
        'Percentage (%)': value_counts.round(2)
    }).reset_index()
    formatted_data.columns = [column.capitalize(), 'Count', 'Percentage (%)']
    return formatted_data

# Main content based on selected page
if page == "📄 Upload Your Data":
    st.header("📄 Data Upload")
    st.info("Upload your customer churn CSV file or use the sample data provided.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Your Data")
        uploaded_data = upload_data()
        if uploaded_data is not None:
            st.session_state.data = uploaded_data
            st.success("✅ Data updated successfully!")
            st.write("**Data shape:**", uploaded_data.shape)
            st.dataframe(uploaded_data.head())
    
    with col2:
        st.subheader("📊 Use Sample Data")
        if st.button("🔄 Load Sample Data"):
            st.session_state.data = load_sample_data()
            st.success("✅ Sample data loaded!")
        
        st.write("**Current dataset info:**")
        st.write(f"Shape: {st.session_state.data.shape}")
        st.write(f"Columns: {len(st.session_state.data.columns)}")

elif page == "🏠 Home & Data Overview":
    st.header("🏠 Data Overview")
    
    df = st.session_state.data.copy()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df)
        st.metric("👥 Total Customers", f"{total_customers:,}")
    
    with col2:
        churn_count = (df['Attrition_Flag'] == 'Attrited Customer').sum()
        churn_rate = (churn_count / total_customers) * 100
        st.metric("📉 Churn Rate", f"{churn_rate:.1f}%")
    
    with col3:
        avg_age = df['Customer_Age'].mean()
        st.metric("👤 Average Age", f"{avg_age:.1f} years")
    
    with col4:
        avg_credit = df['Credit_Limit'].mean()
        st.metric("💳 Avg Credit Limit", f"${avg_credit:,.0f}")
    
    # Data shape and info
    st.subheader("📋 Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**")
        st.info(f"**Rows:** {df.shape[0]:,} | **Columns:** {df.shape[1]}")
        
        st.write("**Missing Values:**")
        missing_data = df.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values found")
        else:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Percentage': (missing_data.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0])
    
    with col2:
        st.write("**Data Types:**")
        dtypes_df = pd.DataFrame({
            'Column': df.dtypes.index,
            'Data Type': df.dtypes.values
        })
        st.dataframe(dtypes_df)
    
    # Sample data
    st.subheader("🔍 Data Sample")
    st.dataframe(df.head(10))
    
    # Basic statistics
    st.subheader("📊 Descriptive Statistics")
    
    # Numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numerical_cols:
        st.write("**Numerical Features:**")
        st.dataframe(df[numerical_cols].describe().round(2))
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        st.write("**Categorical Features:**")
        for col in categorical_cols:
            with st.expander(f"📊 {col} Distribution"):
                value_counts_df = display_value_counts(df, col)
                st.dataframe(value_counts_df)

elif page == "📊 Data Exploration":
    st.header("📊 Data Exploration")
    
    df = st.session_state.data.copy()
    
    # Separate numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    st.subheader("🔢 Numerical Features Analysis")
    
    if numerical_columns:
        # Display numerical statistics
        st.write("**Statistical Summary:**")
        stats_df = df[numerical_columns].describe().round(2)
        st.dataframe(stats_df)
        
        # Correlation analysis
        st.write("**Correlation Analysis:**")
        correlation_matrix = df[numerical_columns].corr()
        
        # Create correlation heatmap
        fig_corr = px.imshow(
            correlation_matrix.round(2),
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu_r"
        )
        fig_corr.update_layout(width=800, height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature distributions by churn
        st.write("**Feature Analysis by Churn Status:**")
        selected_feature = st.selectbox("Select a numerical feature:", numerical_columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            fig_dist = px.histogram(
                df, x=selected_feature, color='Attrition_Flag',
                title=f'Distribution of {selected_feature}',
                nbins=30,
                opacity=0.7
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = px.box(
                df, x='Attrition_Flag', y=selected_feature,
                title=f'{selected_feature} by Churn Status',
                color='Attrition_Flag'
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Statistical comparison
        existing_customers = df[df['Attrition_Flag'] == 'Existing Customer'][selected_feature]
        churned_customers = df[df['Attrition_Flag'] == 'Attrited Customer'][selected_feature]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Existing Customers (Mean)", f"{existing_customers.mean():.2f}")
        with col2:
            st.metric("Churned Customers (Mean)", f"{churned_customers.mean():.2f}")
        with col3:
            difference = churned_customers.mean() - existing_customers.mean()
            st.metric("Difference", f"{difference:.2f}")
    
    st.subheader("📊 Categorical Features Analysis")
    
    if categorical_columns:
        for col in categorical_columns:
            with st.expander(f"📈 {col} Analysis"):
                
                # Value counts
                value_counts_df = display_value_counts(df, col)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Value Counts:**")
                    st.dataframe(value_counts_df)
                
                with col2:
                    # Pie chart
                    fig_pie = px.pie(
                        value_counts_df, 
                        values='Count', 
                        names=col.capitalize(),
                        title=f'{col} Distribution'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Churn analysis by category
                churn_by_category = pd.crosstab(df[col], df['Attrition_Flag'], normalize='index') * 100
                churn_by_category = churn_by_category.round(2)
                
                st.write("**Churn Rate by Category:**")
                st.dataframe(churn_by_category)
                
                # Stacked bar chart
                fig_bar = px.bar(
                    df, x=col, color='Attrition_Flag',
                    title=f'Customer Distribution by {col} and Churn Status',
                    barmode='stack'
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)

elif page == "📈 Data Visualization":
    st.header("📈 Data Visualization")
    
    df = st.session_state.data.copy()
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Overall churn distribution
    st.subheader("🎯 Churn Distribution Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn pie chart
        churn_counts = df['Attrition_Flag'].value_counts()
        fig_pie = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Customer Churn Distribution",
            color_discrete_map={
                'Existing Customer': '#2E8B57',
                'Attrited Customer': '#DC143C'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Churn bar chart
        fig_bar = px.bar(
            x=churn_counts.index,
            y=churn_counts.values,
            title="Customer Count by Status",
            color=churn_counts.index,
            color_discrete_map={
                'Existing Customer': '#2E8B57',
                'Attrited Customer': '#DC143C'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Multi-feature analysis
    st.subheader("🔍 Multi-Feature Analysis")
    
    viz_type = st.selectbox("Select Visualization Type:", [
        "Distribution Plots", "Box Plots", "Scatter Plots", "Feature Comparison"
    ])
    
    if viz_type == "Distribution Plots":
        st.write("**Distribution of Numerical Features by Churn Status**")
        
        # Create subplots for distributions
        selected_features = st.multiselect(
            "Select features to visualize:", 
            numerical_columns, 
            default=numerical_columns[:4]
        )
        
        if selected_features:
            for feature in selected_features:
                fig_hist = px.histogram(
                    df, x=feature, color='Attrition_Flag',
                    title=f'Distribution of {feature}',
                    nbins=30, opacity=0.7,
                    marginal="box"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    
    elif viz_type == "Box Plots":
        st.write("**Box Plots for Feature Comparison**")
        
        # Create box plots
        selected_features = st.multiselect(
            "Select features for box plots:", 
            numerical_columns, 
            default=numerical_columns[:4]
        )
        
        if selected_features:
            for feature in selected_features:
                fig_box = px.box(
                    df, x='Attrition_Flag', y=feature,
                    title=f'{feature} by Churn Status',
                    color='Attrition_Flag'
                )
                st.plotly_chart(fig_box, use_container_width=True)
    
    elif viz_type == "Scatter Plots":
        st.write("**Scatter Plot Analysis**")
        
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("Select X-axis feature:", numerical_columns)
        with col2:
            y_feature = st.selectbox("Select Y-axis feature:", numerical_columns)
        
        if x_feature and y_feature:
            fig_scatter = px.scatter(
                df, x=x_feature, y=y_feature, 
                color='Attrition_Flag',
                title=f'{y_feature} vs {x_feature}',
                opacity=0.6
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    elif viz_type == "Feature Comparison":
        st.write("**Mean Feature Comparison by Churn Status**")
        
        # Calculate means by churn status
        feature_means = df.groupby('Attrition_Flag')[numerical_columns].mean()
        
        selected_features = st.multiselect(
            "Select features to compare:", 
            numerical_columns, 
            default=numerical_columns[:6]
        )
        
        if selected_features:
            comparison_data = feature_means[selected_features].T
            
            fig_comparison = px.bar(
                comparison_data,
                title="Mean Feature Values by Churn Status",
                barmode='group'
            )
            fig_comparison.update_layout(
                xaxis_title="Features",
                yaxis_title="Mean Values",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Show the data table
            st.write("**Mean Values Table:**")
            st.dataframe(comparison_data.round(2))

elif page == "🤖 Machine Learning Models":
    st.header("🤖 Machine Learning Models")
    
    df = st.session_state.data.copy()
    
    # Data preprocessing section
    st.subheader("🔧 Data Preprocessing")
    
    with st.expander("View Preprocessing Steps"):
        st.write("""
        **Preprocessing Steps Applied:**
        1. Remove unknown values from categorical features
        2. Encode target variable (Attrited Customer = 1, Existing Customer = 0)
        3. One-hot encoding for Gender
        4. Ordinal encoding for Income and Card Category
        5. One-hot encoding for Education and Marital Status
        6. Standard scaling for numerical features
        7. SMOTE for handling class imbalance
        """)
    
    # Preprocessing function
    @st.cache_data
    def preprocess_data(data):
        # Remove unknown values
        data_clean = data[
            (data['Income_Category'] != 'Unknown') & 
            (data['Education_Level'] != 'Unknown') & 
            (data['Marital_Status'] != 'Unknown')
        ].copy()
        
        # Encode target variable
        mapping = {'Attrited Customer': 1, 'Existing Customer': 0}
        data_clean['Attrition_Flag'] = data_clean['Attrition_Flag'].replace(mapping)
        
        # Separate features and target
        X = data_clean.drop("Attrition_Flag", axis=1)
        y = data_clean["Attrition_Flag"]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # One-hot encoding for Gender
        X_train = pd.get_dummies(X_train, columns=['Gender'], drop_first=False)
        X_test = pd.get_dummies(X_test, columns=['Gender'], drop_first=False)
        
        # Ordinal encoding for Income
        income_categories = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']
        income_encoder = OrdinalEncoder(categories=[income_categories])
        X_train['Income_Category'] = income_encoder.fit_transform(X_train[['Income_Category']])
        X_test['Income_Category'] = income_encoder.transform(X_test[['Income_Category']])
        
        # Ordinal encoding for Card Category
        card_encoder = OrdinalEncoder(categories=[['Blue', 'Silver', 'Gold', 'Platinum']])
        X_train['Card_Category'] = card_encoder.fit_transform(X_train[['Card_Category']])
        X_test['Card_Category'] = card_encoder.transform(X_test[['Card_Category']])
        
        # One-hot encoding for Education and Marital Status
        education_train = pd.get_dummies(X_train['Education_Level'], prefix='Education')
        education_test = pd.get_dummies(X_test['Education_Level'], prefix='Education')
        marital_train = pd.get_dummies(X_train['Marital_Status'], prefix='Marital')
        marital_test = pd.get_dummies(X_test['Marital_Status'], prefix='Marital')
        
        # Combine encoded features
        X_train = pd.concat([X_train, education_train, marital_train], axis=1)
        X_test = pd.concat([X_test, education_test, marital_test], axis=1)
        X_train = X_train.drop(['Education_Level', 'Marital_Status'], axis=1)
        X_test = X_test.drop(['Education_Level', 'Marital_Status'], axis=1)
        
        # Align columns
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        
        # Scale numerical features
        numerical_columns = data_clean.select_dtypes(include=['number']).columns.tolist()
        if 'Attrition_Flag' in numerical_columns:
            numerical_columns.remove('Attrition_Flag')
        
        scaler = StandardScaler()
        X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
        
        return X_train, X_test, y_train, y_test
    
    # Preprocess the data
    try:
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Show class distribution before SMOTE
        st.write("**Class Distribution (Before SMOTE):**")
        class_dist = pd.Series(y_train).value_counts()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Existing Customers", class_dist[0])
        with col2:
            st.metric("Churned Customers", class_dist[1])
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        st.write("**Class Distribution (After SMOTE):**")
        class_dist_balanced = pd.Series(y_train_balanced).value_counts()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Existing Customers", class_dist_balanced[0])
        with col2:
            st.metric("Churned Customers", class_dist_balanced[1])
        
        # Model selection
        st.subheader("🎯 Model Training & Evaluation")
        
        model_type = st.selectbox("Select Model:", [
            "Random Forest", "Logistic Regression", "KNN", 
            "Decision Tree", "SVM"
        ])
        
        if st.button("🚀 Train Model"):
            
            # Progress bar
            progress_bar = st.progress(0)
            
            # Initialize model based on selection
            if model_type == "Random Forest":
                model = RandomForestClassifier(
                    random_state=42,
                    class_weight={0:1, 1:5},
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt'
                )
            elif model_type == "Logistic Regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif model_type == "KNN":
                model = KNeighborsClassifier(n_neighbors=5)
            elif model_type == "Decision Tree":
                model = DecisionTreeClassifier(
                    random_state=42,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2
                )
            elif model_type == "SVM":
                model = SVC(random_state=42, probability=True)
            
            progress_bar.progress(25)
            
            # Train model
            model.fit(X_train_balanced, y_train_balanced)
            progress_bar.progress(50)
            
            # Make predictions
            y_pred_train = model.predict(X_train_balanced)
            y_pred_test = model.predict(X_test)
            progress_bar.progress(75)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train_balanced, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            train_recall = recall_score(y_train_balanced, y_pred_train, average='macro')
            test_recall = recall_score(y_test, y_pred_test, average='macro')
            
            progress_bar.progress(100)
            
            # Display results
            st.subheader("📊 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train Accuracy", f"{train_accuracy:.3f}")
            with col2:
                st.metric("Test Accuracy", f"{test_accuracy:.3f}")
            with col3:
                st.metric("Train Recall", f"{train_recall:.3f}")
            with col4:
                st.metric("Test Recall", f"{test_recall:.3f}")
            
            # Classification report
            st.write("**Classification Report (Test Data):**")
            report = classification_report(y_test, y_pred_test, output_dict=True)
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df)
            
            # Confusion Matrix
            st.write("**Confusion Matrix:**")
            cm = confusion_matrix(y_test, y_pred_test)
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                x=['Existing', 'Churned'],
                y=['Existing', 'Churned']
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                st.write("**Feature Importance:**")
                feature_importance = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(
                    feature_importance.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 15 Feature Importance"
                )
                fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Show feature importance table
                st.dataframe(feature_importance.head(10))
        
    except Exception as e:
        st.error(f"❌ Error in preprocessing: {str(e)}")
        st.write("Please make sure your data has the required columns.")

elif page == "🔧 Model Optimization":
    st.header("🔧 Model Optimization")
    
    df = st.session_state.data.copy()
    
    st.info("This section performs hyperparameter tuning using GridSearchCV to find the best model parameters.")
    
    # Preprocessing for optimization
    try:
        # Use the same preprocessing function
        def preprocess_data_opt(data):
            data_clean = data[
                (data['Income_Category'] != 'Unknown') & 
                (data['Education_Level'] != 'Unknown') & 
                (data['Marital_Status'] != 'Unknown')
            ].copy()
            
            mapping = {'Attrited Customer': 1, 'Existing Customer': 0}
            data_clean['Attrition_Flag'] = data_clean['Attrition_Flag'].replace(mapping)
            
            X = data_clean.drop("Attrition_Flag", axis=1)
            y = data_clean["Attrition_Flag"]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Encoding steps (same as before)
            X_train = pd.get_dummies(X_train, columns=['Gender'], drop_first=False)
            X_test = pd.get_dummies(X_test, columns=['Gender'], drop_first=False)
            
            income_categories = ['Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +']
            income_encoder = OrdinalEncoder(categories=[income_categories])
            X_train['Income_Category'] = income_encoder.fit_transform(X_train[['Income_Category']])
            X_test['Income_Category'] = income_encoder.transform(X_test[['Income_Category']])
            
            card_encoder = OrdinalEncoder(categories=[['Blue', 'Silver', 'Gold', 'Platinum']])
            X_train['Card_Category'] = card_encoder.fit_transform(X_train[['Card_Category']])
            X_test['Card_Category'] = card_encoder.transform(X_test[['Card_Category']])
            
            education_train = pd.get_dummies(X_train['Education_Level'], prefix='Education')
            education_test = pd.get_dummies(X_test['Education_Level'], prefix='Education')
            marital_train = pd.get_dummies(X_train['Marital_Status'], prefix='Marital')
            marital_test = pd.get_dummies(X_test['Marital_Status'], prefix='Marital')
            
            X_train = pd.concat([X_train, education_train, marital_train], axis=1)
            X_test = pd.concat([X_test, education_test, marital_test], axis=1)
            X_train = X_train.drop(['Education_Level', 'Marital_Status'], axis=1)
            X_test = X_test.drop(['Education_Level', 'Marital_Status'], axis=1)
            
            X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
            
            numerical_columns = data_clean.select_dtypes(include=['number']).columns.tolist()
            if 'Attrition_Flag' in numerical_columns:
                numerical_columns.remove('Attrition_Flag')
            
            scaler = StandardScaler()
            X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
            X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
            return X_train, X_test, y_train, y_test
        
        # Model selection for optimization
        optimization_model = st.selectbox("Select Model for Optimization:", [
            "Random Forest", "Logistic Regression", "KNN", "Decision Tree", "SVM"
        ])
        
        # Define parameter grids
        param_grids = {
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            "Logistic Regression": {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            "SVM": {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            },
            "KNN": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            "Decision Tree": {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'criterion': ['gini', 'entropy']
            }
        }
        
        st.write(f"**Parameter Grid for {optimization_model}:**")
        st.json(param_grids[optimization_model])
        
        if st.button("🔍 Start Optimization"):
            
            # Preprocess data
            X_train, X_test, y_train, y_test = preprocess_data_opt(df)
            
            # Initialize model
            if optimization_model == "Random Forest":
                model = RandomForestClassifier(random_state=42)
            elif optimization_model == "Logistic Regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif optimization_model == "KNN":
                model = KNeighborsClassifier()
            elif optimization_model == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
            elif optimization_model == "SVM":
                model = SVC(random_state=42)
            
            # Progress tracking
            with st.spinner(f'Optimizing {optimization_model}... This may take a few minutes.'):
                
                # GridSearchCV
                grid_search = GridSearchCV(
                    model, 
                    param_grids[optimization_model], 
                    cv=5, 
                    scoring='accuracy', 
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                
                # Make predictions
                y_pred_test = best_model.predict(X_test)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                test_recall = recall_score(y_test, y_pred_test, average='macro')
                
                # Display results
                st.success("✅ Optimization completed!")
                
                st.subheader("🏆 Best Model Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Best CV Score", f"{grid_search.best_score_:.4f}")
                    st.metric("Test Accuracy", f"{test_accuracy:.4f}")
                with col2:
                    st.metric("Test Recall (Macro)", f"{test_recall:.4f}")
                
                st.write("**Best Parameters:**")
                st.json(grid_search.best_params_)
                
                st.write("**Classification Report:**")
                report = classification_report(y_test, y_pred_test, output_dict=True)
                report_df = pd.DataFrame(report).transpose().round(4)
                st.dataframe(report_df)
                
                st.write("**Confusion Matrix:**")
                cm = confusion_matrix(y_test, y_pred_test)
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    aspect="auto",
                    title="Confusion Matrix - Optimized Model",
                    labels=dict(x="Predicted", y="Actual"),
                    x=['Existing', 'Churned'],
                    y=['Existing', 'Churned']
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
                # Cross-validation scores
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
                st.write("**Cross-Validation Results:**")
                cv_df = pd.DataFrame({
                    'Fold': range(1, 6),
                    'Accuracy': cv_scores
                })
                st.dataframe(cv_df.round(4))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean CV Accuracy", f"{cv_scores.mean():.4f}")
                with col2:
                    st.metric("CV Std Deviation", f"{cv_scores.std():.4f}")
                
                # Feature importance for tree-based models
                if hasattr(best_model, 'feature_importances_'):
                    st.write("**Feature Importance (Optimized Model):**")
                    feature_importance = pd.DataFrame({
                        'Feature': X_train.columns,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig_importance = px.bar(
                        feature_importance.head(15),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Top 15 Features - Optimized Model"
                    )
                    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_importance, use_container_width=True)
        
        # Model comparison section
        st.subheader("📊 Model Comparison")
        
        if st.button("🔄 Compare All Models"):
            
            X_train, X_test, y_train, y_test = preprocess_data_opt(df)
            
            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "SVM": SVC(random_state=42)
            }
            
            results = []
            
            progress_bar = st.progress(0)
            
            for i, (name, model) in enumerate(models.items()):
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred, average='macro')
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                results.append({
                    'Model': name,
                    'Test Accuracy': accuracy,
                    'Test Recall (Macro)': recall,
                    'CV Mean': cv_scores.mean(),
                    'CV Std': cv_scores.std()
                })
                
                progress_bar.progress((i + 1) / len(models))
            
            # Display comparison
            results_df = pd.DataFrame(results).round(4)
            st.dataframe(results_df)
            
            # Visualization
            fig_comparison = px.bar(
                results_df, 
                x='Model', 
                y='Test Accuracy',
                title="Model Accuracy Comparison",
                color='Test Accuracy',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Best model highlight
            best_model_idx = results_df['Test Accuracy'].idxmax()
            best_model_name = results_df.loc[best_model_idx, 'Model']
            best_accuracy = results_df.loc[best_model_idx, 'Test Accuracy']
            
            st.success(f"🏆 Best performing model: **{best_model_name}** with accuracy: **{best_accuracy:.4f}**")
        
    except Exception as e:
        st.error(f"❌ Error in optimization: {str(e)}")
        st.write("Please make sure your data has the required columns for optimization.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Customer Churn Analysis**")
    st.markdown("Built with Streamlit & Python")

with col2:
    if st.button("📥 Download Sample Data"):
        csv = st.session_state.data.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name="customer_churn_data.csv",
            mime="text/csv"
        )

with col3:
    st.markdown("**🔗 Quick Actions**")
    if st.button("🔄 Refresh Data"):
        st.session_state.data = load_sample_data()
        st.success("✅ Data refreshed!")
        st.rerun()

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Dataset Info")
st.sidebar.write(f"**Rows:** {st.session_state.data.shape[0]:,}")
st.sidebar.write(f"**Columns:** {st.session_state.data.shape[1]}")

churn_rate = (st.session_state.data['Attrition_Flag'] == 'Attrited Customer').mean() * 100
st.sidebar.write(f"**Churn Rate:** {churn_rate:.1f}%")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.write("""
This application provides comprehensive customer churn analysis including:
- **Data Exploration**: Statistical analysis and insights
- **Visualization**: Interactive charts and plots  
- **Machine Learning**: Predictive modeling with multiple algorithms
- **Optimization**: Hyperparameter tuning and model comparison
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Features")
st.sidebar.write("""
✅ Interactive data exploration  
✅ Real-time visualizations  
✅ Multiple ML algorithms  
✅ Model optimization  
✅ Performance comparison  
✅ Export functionality  
""")
