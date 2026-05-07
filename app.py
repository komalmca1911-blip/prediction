# app.py
# Save this file as 'app.py' and run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="AI Finance Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS FOR DARK FINTECH THEME ====================
st.markdown("""
<style>
    /* Main background and text */
    .stApp {
        background-color: #0a0e1a;
    }
    
    /* Main container */
    .main {
        background-color: #0a0e1a;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #10b981 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937 !important;
    }
    
    [data-testid="stSidebar"] .st-emotion-cache-1wrcr25 {
        background-color: #111827 !important;
    }
    
    /* Input labels */
    .st-emotion-cache-1y4p8pa, .st-emotion-cache-16c6r5l {
        color: #9ca3af !important;
        font-weight: 500 !important;
    }
    
    /* Number input styling */
    .stNumberInput input, .stSelectbox input, .stTextInput input {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        color: white !important;
        border-radius: 8px !important;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #10b981 !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
    }
    
    .stButton button:hover {
        background-color: #059669 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #374151;
    }
    
    [data-testid="stMetric"] label {
        color: #9ca3af !important;
    }
    
    [data-testid="stMetric"] .st-emotion-cache-1jri0rr {
        color: #10b981 !important;
    }
    
    /* Alert/warning boxes */
    .alert-orange {
        background: linear-gradient(135deg, #78350f 0%, #451a03 100%);
        border-left: 4px solid #f97316;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-emerald {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .alert-red {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Divider */
    hr {
        border-color: #374151 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD YOUR ACTUAL MODEL ====================
@st.cache_resource
def load_model_and_encoders():
    """
    Load the trained Random Forest model and LabelEncoders
    This uses the exact model trained from your dataset
    """
    
    # Since you trained the model in the same session, we'll recreate it with your actual data
    # First, load your dataset
    try:
        # Try to load your actual CSV file
        df = pd.read_csv('data_set.csv')
    except FileNotFoundError:
        # If not found, try relative path
        try:
            df = pd.read_csv('data_set.csv')
        except FileNotFoundError:
            st.error("❌ data_set.csv not found! Please make sure the file is in the same directory.")
            return None, None, None, None
    
    # Standardize capitalization exactly like your original code
    df['Occupation'] = df['Occupation'].str.strip().str.capitalize()
    df['City_Tier'] = df['City_Tier'].str.strip().str.capitalize()
    
    # Create and fit LabelEncoders exactly as in your code
    le_occ = LabelEncoder()
    df['Occupation_Enc'] = le_occ.fit_transform(df['Occupation'])
    
    le_city = LabelEncoder()
    df['City_Tier_Enc'] = le_city.fit_transform(df['City_Tier'])
    
    # Define features exactly as in your model
    features = ['Income', 'Age', 'Dependents', 'Occupation_Enc', 'City_Tier_Enc', 'Rent', 
                'Loan_Repayment', 'Insurance', 'Groceries', 'Transport', 'Eating_Out', 
                'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']
    target = 'Desired_Savings'
    
    # Check if all required columns exist
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {missing_cols}")
        return None, None, None, None
    
    if target not in df.columns:
        st.error(f"Target column '{target}' not found in dataset")
        return None, None, None, None
    
    X = df[features]
    y = df[target]
    
    # Train the Random Forest model with the same parameters as your code
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate and store model accuracy
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return model, le_occ, le_city, df, train_score, test_score

# Load the model
model, le_occ, le_city, df, train_score, test_score = load_model_and_encoders()

if model is None:
    st.stop()  # Stop execution if model couldn't be loaded

# ==================== HELPER FUNCTIONS ====================

def predict_savings(user_data):
    """
    Make prediction using your actual trained model
    """
    # Define features in the exact order your model expects
    features = ['Income', 'Age', 'Dependents', 'Occupation_Enc', 'City_Tier_Enc', 'Rent', 
                'Loan_Repayment', 'Insurance', 'Groceries', 'Transport', 'Eating_Out', 
                'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']
    
    # Create DataFrame with the exact feature order
    input_df = pd.DataFrame([[user_data[feature] for feature in features]], columns=features)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    return max(0, prediction)  # Ensure non-negative savings

def calculate_financial_health(income, total_expenses, predicted_savings):
    """Calculate financial health metrics and return insights"""
    savings_rate = predicted_savings / income if income > 0 else 0
    expense_ratio = total_expenses / income if income > 0 else 1
    disposable_income = income - total_expenses
    
    insights = []
    
    if total_expenses > income:
        insights.append(("🔴 CRITICAL", f"You are spending ₹{total_expenses - income:,.0f} more than you earn! Immediate action needed.", "red"))
    elif expense_ratio > 0.8:
        insights.append(("⚠️ WARNING", "You're spending over 80% of your income. Consider reducing expenses to increase savings.", "orange"))
    elif expense_ratio < 0.5:
        insights.append(("✅ EXCELLENT", f"Great job! You're saving {savings_rate*100:.1f}% of your income. Keep it up!", "emerald"))
    else:
        insights.append(("ℹ️ FAIR", "Your spending is within a reasonable range. Try to increase your savings rate to 20%+.", "emerald"))
    
    return insights, savings_rate, expense_ratio, disposable_income

def get_category_insights(expenses, income):
    """Generate category-specific advice based on your original logic"""
    advice = []
    
    # Mirroring your original logic
    if expenses.get('Eating_Out', 0) > income * 0.15:
        advice.append(("🍽️ Dining Out", "Your dining expenses exceed 15% of income. This is your biggest saving opportunity - consider cooking more at home!", "orange"))
    elif expenses.get('Eating_Out', 0) > income * 0.1:
        advice.append(("🍽️ Dining Out", "Your dining expenses are moderate. Try reducing by 20% to boost savings.", "emerald"))
    
    if expenses.get('Utilities', 0) > income * 0.10:
        advice.append(("💡 Utilities", "Your utility bills are high. Look for ways to reduce power or data costs. Consider energy-efficient appliances.", "orange"))
    
    if expenses.get('Entertainment', 0) > income * 0.1:
        advice.append(("🎬 Entertainment", "Entertainment spending is high. Try free alternatives or budget-friendly options.", "orange"))
    
    if expenses.get('Rent', 0) > income * 0.4:
        advice.append(("🏠 Housing", "Your rent is over 40% of income. Consider finding cheaper accommodation or a roommate.", "orange"))
    elif expenses.get('Rent', 0) > income * 0.3:
        advice.append(("🏠 Housing", "Your rent is within 30-40% of income, which is acceptable but could be optimized.", "emerald"))
    
    if expenses.get('Transport', 0) > income * 0.15:
        advice.append(("🚗 Transport", "Transport costs are high. Consider public transport, carpooling, or cycling.", "orange"))
    
    if expenses.get('Groceries', 0) > income * 0.2:
        advice.append(("🛒 Groceries", "Grocery spending is high. Plan meals, use shopping lists, and avoid waste.", "orange"))
    
    return advice

def create_expense_pie_chart(expenses):
    """Create a pie chart for expense breakdown"""
    filtered = {k: v for k, v in expenses.items() if v > 0}
    
    if not filtered:
        return None
    
    df_pie = pd.DataFrame({
        'Category': list(filtered.keys()),
        'Amount': list(filtered.values())
    })
    
    colors = ['#10b981', '#34d399', '#059669', '#047857', '#065f46', '#064e3b', '#022c22']
    
    fig = px.pie(
        df_pie, 
        values='Amount', 
        names='Category',
        title="Expense Breakdown",
        color_discrete_sequence=colors,
        hole=0.4
    )
    
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font={"color": "#9ca3af"},
        title_font={"color": "#10b981"},
        legend={"font": {"color": "#9ca3af"}}
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+percent+value',
        marker=dict(line=dict(color="#111827", width=2))
    )
    
    return fig

def create_gauge_chart(value, max_value=100, title="Health Score"):
    """Create a gauge chart using plotly"""
    percentage = min(100, (value / max_value) * 100) if max_value > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        title={"text": title, "font": {"color": "#9ca3af", "size": 14}},
        number={"font": {"color": "#10b981", "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#374151"},
            "bar": {"color": "#10b981", "thickness": 0.3},
            "bgcolor": "#1f2937",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#451a03"},
                {"range": [50, 75], "color": "#78350f"},
                {"range": [75, 100], "color": "#064e3b"}
            ],
            "threshold": {
                "line": {"color": "#f97316", "width": 4},
                "thickness": 0.75,
                "value": percentage
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(t=30, b=20, l=20, r=20),
        paper_bgcolor="#111827",
        font={"color": "#9ca3af"}
    )
    
    return fig

# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>💰 AI Personal Finance Tracker</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9ca3af;'>Powered by Random Forest Machine Learning</p>", unsafe_allow_html=True)
    
    # Display model info in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"""
        <div style='background-color: #1f2937; border-radius: 8px; padding: 0.75rem; margin-bottom: 1rem;'>
            <p style='color: #9ca3af; font-size: 0.8rem; margin: 0;'>🤖 Model: Random Forest</p>
            <p style='color: #10b981; font-size: 0.8rem; margin: 0;'>📊 R² Score: {test_score*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    st.markdown("---")
    
    # Sidebar - User Inputs
    with st.sidebar:
        st.markdown("<h3 style='color: #10b981;'>👤 Profile Information</h3>", unsafe_allow_html=True)
        
        income = st.number_input("💰 Monthly Income (₹)", min_value=0, value=50000, step=5000, help="Your total monthly income")
        age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30, step=1)
        
        # Get occupation options from your trained encoder
        occupation_options = list(le_occ.classes_)
        occupation = st.selectbox(
            "💼 Occupation",
            occupation_options,
            help="Select your current employment status"
        )
        
        # Get city tier options from your trained encoder
        city_options = list(le_city.classes_)
        city_tier = st.selectbox(
            "🏙️ City Tier",
            city_options,
            help="Select your city tier"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #10b981;'>📋 Monthly Expenses</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            rent = st.number_input("🏠 Rent/Housing", min_value=0, value=15000, step=1000)
            groceries = st.number_input("🛒 Groceries", min_value=0, value=8000, step=500)
            transport = st.number_input("🚗 Transport", min_value=0, value=5000, step=500)
            eating_out = st.number_input("🍽️ Eating Out", min_value=0, value=4000, step=500)
        with col2:
            entertainment = st.number_input("🎬 Entertainment", min_value=0, value=3000, step=500)
            utilities = st.number_input("💡 Utilities", min_value=0, value=4000, step=500)
            misc = st.number_input("📦 Miscellaneous", min_value=0, value=3000, step=500)
        
        # Additional fields that your model expects (with default values)
        with st.expander("🔧 Additional Expenses (Optional)"):
            dependents = st.number_input("👨‍👩‍👧‍👦 Number of Dependents", min_value=0, value=0, step=1)
            loan_repayment = st.number_input("🏦 Loan Repayment", min_value=0, value=0, step=1000)
            insurance = st.number_input("🛡️ Insurance", min_value=0, value=0, step=1000)
            healthcare = st.number_input("🏥 Healthcare", min_value=0, value=0, step=500)
            education = st.number_input("📚 Education", min_value=0, value=0, step=1000)
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 PREDICT SAVINGS GOAL", use_container_width=True)
    
    # Prepare expense dictionary
    expenses = {
        'Rent': rent,
        'Groceries': groceries,
        'Transport': transport,
        'Eating_Out': eating_out,
        'Entertainment': entertainment,
        'Utilities': utilities,
        'Misc': misc
    }
    
    total_expenses = sum(expenses.values())
    
    # Encode categorical variables using your trained encoders
    occ_enc = le_occ.transform([occupation])[0]
    city_enc = le_city.transform([city_tier])[0]
    
    # Prepare user data for prediction exactly as your model expects
    user_data = {
        'Income': income,
        'Age': age,
        'Dependents': dependents,
        'Occupation_Enc': occ_enc,
        'City_Tier_Enc': city_enc,
        'Rent': rent,
        'Loan_Repayment': loan_repayment,
        'Insurance': insurance,
        'Groceries': groceries,
        'Transport': transport,
        'Eating_Out': eating_out,
        'Entertainment': entertainment,
        'Utilities': utilities,
        'Healthcare': healthcare,
        'Education': education,
        'Miscellaneous': misc
    }
    
    # Default display or prediction
    if predict_btn or 'last_prediction' in st.session_state:
        if predict_btn:
            with st.spinner("🧠 AI is analyzing your financial profile..."):
                predicted_savings = predict_savings(user_data)
                st.session_state.last_prediction = predicted_savings
                st.session_state.last_income = income
                st.session_state.last_expenses = total_expenses
                st.session_state.last_expense_dict = expenses.copy()
        else:
            predicted_savings = st.session_state.last_prediction
            income = st.session_state.last_income
            total_expenses = st.session_state.last_expenses
            expenses = st.session_state.last_expense_dict
        
        # Calculate financial health
        insights, savings_rate, expense_ratio, disposable_income = calculate_financial_health(
            income, total_expenses, predicted_savings
        )
        
        # Alert if spending exceeds income (mirroring your original logic)
        if total_expenses > income:
            deficit = total_expenses - income
            st.markdown(f"""
            <div class='alert-red'>
                <strong>🔴 ALERT:</strong> You are spending ₹{deficit:,.0f} more than you earn!
            </div>
            """, unsafe_allow_html=True)
        
        # Main content area with results
        col1, col2, col3 = st.columns([1, 1.5, 1])
        
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #064e3b 0%, #022c22 100%); 
                        border-radius: 24px; padding: 2rem; text-align: center; 
                        border: 1px solid #10b981; margin: 1rem 0;'>
                <p style='color: #9ca3af; margin-bottom: 0.5rem; font-size: 0.9rem;'>🎯 AI RECOMMENDED SAVINGS GOAL</p>
                <h1 style='color: #10b981; font-size: 3.5rem; margin: 0;'>₹{predicted_savings:,.0f}</h1>
                <p style='color: #6ee7b7; margin-top: 0.5rem; font-size: 0.85rem;'>
                    {savings_rate*100:.1f}% of your income
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Key metrics row
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("💰 Monthly Income", f"₹{income:,.0f}")
        with metric_col2:
            st.metric("📊 Total Expenses", f"₹{total_expenses:,.0f}", 
                     delta=f"{((expense_ratio-1)*100):.0f}% over" if expense_ratio > 1 else f"{expense_ratio*100:.0f}% of income")
        with metric_col3:
            st.metric("💵 Disposable Income", f"₹{disposable_income:,.0f}")
        with metric_col4:
            st.metric("🎯 Savings Rate", f"{savings_rate*100:.1f}%")
        
        st.markdown("---")
        
        # Charts row
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("<h3 style='text-align: center;'>📊 Expense Breakdown</h3>", unsafe_allow_html=True)
            pie_chart = create_expense_pie_chart(expenses)
            if pie_chart:
                st.plotly_chart(pie_chart, use_container_width=True)
            else:
                st.info("No expense data to display. Add some expenses to see the breakdown.")
        
        with chart_col2:
            st.markdown("<h3 style='text-align: center;'>📈 Financial Health Gauge</h3>", unsafe_allow_html=True)
            # Calculate a health score based on multiple factors
            health_score = min(100, max(0, 100 - (expense_ratio * 60) + (savings_rate * 40)))
            health_score = min(100, max(0, health_score))
            gauge = create_gauge_chart(health_score, 100, "Financial Health")
            st.plotly_chart(gauge, use_container_width=True)
            
            # Progress bar for spending vs income
            spend_percent = min(100, (total_expenses / income) * 100) if income > 0 else 0
            st.markdown(f"""
            <div style='margin-top: 1rem;'>
                <p style='color: #9ca3af; margin-bottom: 0.25rem;'>💳 Spending vs Income</p>
                <div style='background-color: #1f2937; border-radius: 12px; height: 12px; overflow: hidden;'>
                    <div style='width: {spend_percent}%; background-color: {"#ef4444" if spend_percent > 80 else "#f97316" if spend_percent > 60 else "#10b981"}; 
                                height: 100%; border-radius: 12px; transition: width 0.5s ease;'></div>
                </div>
                <p style='color: #d1d5db; margin-top: 0.25rem; font-size: 0.85rem;'>You've spent {spend_percent:.1f}% of your income</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Insights Section
        st.markdown("<h3 style='color: #10b981;'>🔍 Financial Health Dashboard</h3>", unsafe_allow_html=True)
        
        # Main health alert
        for title, message, severity in insights:
            alert_class = "alert-red" if severity == "red" else "alert-orange" if severity == "orange" else "alert-emerald"
            st.markdown(f"""
            <div class='{alert_class}'>
                <strong>{title}</strong> {message}
            </div>
            """, unsafe_allow_html=True)
        
        # Category-specific advice matching your original logic
        category_advice = get_category_insights(expenses, income)
        if category_advice:
            st.markdown("<h4 style='color: #9ca3af; margin-top: 1rem;'>💡 AI-Powered Recommendations</h4>", unsafe_allow_html=True)
            
            advice_cols = st.columns(2)
            for idx, (category, advice, severity) in enumerate(category_advice):
                border_color = "#f97316" if severity == "orange" else "#10b981"
                with advice_cols[idx % 2]:
                    st.markdown(f"""
                    <div style='background-color: #1f2937; border-left: 4px solid {border_color}; 
                                border-radius: 8px; padding: 0.75rem; margin: 0.5rem 0;'>
                        <strong style='color: {border_color};'>{category}</strong><br>
                        <span style='color: #d1d5db; font-size: 0.85rem;'>{advice}</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Smart saving tips
        st.markdown("<h4 style='color: #9ca3af; margin-top: 1rem;'>📌 Smart Saving Tips</h4>", unsafe_allow_html=True)
        
        # Generate tips based on user's data
        tips = []
        if eating_out > income * 0.15:
            tips.append("🍽️ " + "Your dining out is high - try the 50/30/20 rule: 50% needs, 30% wants, 20% savings")
        if predicted_savings < income * 0.2:
            tips.append("💪 " + "Aim to save at least 20% of your income. Consider automating your savings")
        if rent > income * 0.3:
            tips.append("🏠 " + "Housing should ideally be under 30% of income. Consider roommates or relocation")
        if loan_repayment > income * 0.2:
            tips.append("💰 " + "High loan repayment detected. Consider refinancing or debt consolidation")
        if len(tips) < 2:
            tips.append("📱 " + "Use round-up apps that save your spare change automatically")
            tips.append("📊 " + "Track every expense for a month to identify hidden spending leaks")
        
        for tip in tips[:3]:
            st.markdown(f"<p style='color: #d1d5db; margin: 0.25rem 0;'>{tip}</p>", unsafe_allow_html=True)
    
    else:
        # Welcome screen when no prediction yet
        st.markdown("""
        <div style='text-align: center; padding: 3rem 2rem; background: linear-gradient(135deg, #111827 0%, #0a0e1a 100%); 
                    border-radius: 24px; border: 1px solid #1f2937; margin: 2rem 0;'>
            <h2 style='color: #10b981;'>Welcome to Your AI Finance Coach</h2>
            <p style='color: #9ca3af; font-size: 1.1rem; max-width: 600px; margin: 1rem auto;'>
                Enter your financial details in the sidebar and click <strong>"Predict Savings Goal"</strong> 
                to get AI-powered insights and personalized recommendations from your Random Forest model.
            </p>
            <div style='display: flex; justify-content: center; gap: 2rem; margin-top: 2rem; flex-wrap: wrap;'>
                <div><span style='color: #10b981;'>🎯</span> <span style='color: #d1d5db;'>Smart Savings Goals</span></div>
                <div><span style='color: #10b981;'>📊</span> <span style='color: #d1d5db;'>Expense Analytics</span></div>
                <div><span style='color: #10b981;'>🤖</span> <span style='color: #d1d5db;'>ML Predictions</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background: #111827; border-radius: 12px; padding: 1.5rem; text-align: center; border: 1px solid #1f2937;'>
                <h3 style='color: #10b981; margin-bottom: 0.5rem;'>📈 Random Forest Model</h3>
                <p style='color: #9ca3af;'>Trained on your dataset with 100 estimators for accurate savings predictions</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background: #111827; border-radius: 12px; padding: 1.5rem; text-align: center; border: 1px solid #1f2937;'>
                <h3 style='color: #10b981; margin-bottom: 0.5rem;'>🎨 Visual Analytics</h3>
                <p style='color: #9ca3af;'>Interactive pie charts and gauges for instant financial clarity</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style='background: #111827; border-radius: 12px; padding: 1.5rem; text-align: center; border: 1px solid #1f2937;'>
                <h3 style='color: #10b981; margin-bottom: 0.5rem;'>💡 Smart Insights</h3>
                <p style='color: #9ca3af;'>Personalized advice based on your spending patterns</p>
            </div>
            """, unsafe_allow_html=True)

if _name_ == "_main_":
    main()