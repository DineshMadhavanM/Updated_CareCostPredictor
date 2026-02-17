import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import load_model, predict_cost, get_risk_level, get_govt_vs_private_comparison, generate_pdf_report, estimate_accident_injury_cost, get_accident_cost_breakdown, get_government_scheme_recommendations
import os
import base64
import io
import auth_utils
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manual fallback for loading .env if python-dotenv is not available
    if os.path.exists('.env'):
        with open('.env') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    os.environ[k] = v

# Translation Dictionaries
from translations import translations

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

# --- Mobile UI Optimization ---
st.markdown("""
<style>
    /* Premium Modern Dashboard Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 16px;
    }

    /* Main Container Styling */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }

    /* Card-like containers for metrics and inputs */
    div.stMetric, div[data-testid="stMetricValue"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 15px !important;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    div[data-testid="stMetricValue"] {
        color: #2dd4bf !important;
        font-weight: 700 !important;
    }

    /* Customizing Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        width: 100%;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
        border: none;
        color: white;
    }

    /* Input focus effects */
    .stTextInput input:focus, .stNumberInput input:focus, .stSelectbox div[data-baseweb="select"]:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #334155;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        background-color: transparent;
        border: none;
        color: #94a3b8;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
    }

    /* Headings Styling */
    h1 {
        background: linear-gradient(135deg, #2dd4bf 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }

    /* Dividers */
    hr {
        border-color: #334155 !important;
    }

    /* Success/Info boxes */
    div[data-testid="stNotification"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'username' not in st.session_state:
    st.session_state.username = None

if 'email' not in st.session_state:
    st.session_state.email = None

# Helper function to get translation with fallback
def t(key):
    return translations[st.session_state.language].get(key, translations['en'].get(key, key))

# Authentication UI
if not st.session_state.authenticated:
    st.title(t('auth_welcome'))
    
    auth_mode = st.radio("", [t('login'), t('signup')], horizontal=True)
    
    with st.form("auth_form"):
        username_input = st.text_input(t('username'))
        if auth_mode == t('signup'):
            email_input = st.text_input(t('email'))
        password_input = st.text_input(t('password'), type="password")
        if auth_mode == t('signup'):
            confirm_password_input = st.text_input(t('confirm_password'), type="password")
            
        submit_btn = st.form_submit_button(t('login') if auth_mode == t('login') else t('signup'))
        
        if submit_btn:
            if not username_input or not password_input:
                st.error("Please fill in all fields")
            elif auth_mode == t('signup') and (not email_input or not confirm_password_input):
                st.error("Please fill in all fields")
            elif auth_mode == t('signup') and password_input != confirm_password_input:
                st.error(t('passwords_dont_match'))
            elif auth_mode == t('login'):
                success, message, email = auth_utils.login_user(username_input, password_input)
                if success:
                    st.session_state.authenticated = True
                    st.session_state.username = username_input
                    st.session_state.email = email
                    st.success(t('auth_success'))
                    st.rerun()
                else:
                    st.error(t('auth_error'))
            else:
                success, message = auth_utils.sign_up_user(username_input, password_input, email_input)
                if success:
                    st.success(message)
                    st.info(f"Please {t('login')} now.")
                else:
                    st.error(message)
    
    # Language selector also on auth page
    st.markdown("---")
    lang_options = {'English': 'en', 'हिंदी': 'hi', 'தமிழ்': 'ta'}
    # Ensure current language is in options, default to 'en' if not
    current_lang = st.session_state.language if st.session_state.language in lang_options.values() else 'en'
    default_index = list(lang_options.values()).index(current_lang)
    selected_lang_name = st.selectbox(t('language_selector'), options=list(lang_options.keys()), 
                                     index=default_index)
    st.session_state.language = lang_options[selected_lang_name]
    
    st.stop() # Prevents showing the rest of the app

# If authenticated, show logout in sidebar
with st.sidebar:
    st.write(f"👤 {st.session_state.username}")
    st.write(f"📧 {st.session_state.email}") # Display email as well
    if st.button(t('logout')):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.email = None
        st.rerun()
    st.markdown("---")

# Initialize model and data
@st.cache_resource
def initialize_app():
    model_data = load_model()
    
    if os.path.exists('insurance_data.csv'):
        df = pd.read_csv('insurance_data.csv')
    else:
        raise FileNotFoundError("insurance_data.csv not found. Please provide a real insurance dataset.")
    
    return model_data, df

model_data, df = initialize_app()

# Title and description
st.title(t('main_title'))
st.markdown(t('main_description'))

# Sidebar for model info
with st.sidebar:
    # Language selector
    st.header(t('language_selector'))
    language_option = st.selectbox(
        "Select Language", 
        options=['en', 'hi', 'ta'],
        format_func=lambda x: 'English' if x == 'en' else ('हिन्दी' if x == 'hi' else 'தமிழ்'),
        index=0 if st.session_state.language == 'en' else (1 if st.session_state.language == 'hi' else 2),
        key='language_selector',
        label_visibility="collapsed"
    )
    
    if language_option != st.session_state.language:
        st.session_state.language = language_option
        st.rerun()
    
    st.markdown("---")
    st.header(t('model_info'))
    model_type = model_data.get('model_type', 'Random Forest')
    st.metric(t('model_type'), model_type)
    st.metric(t('training_accuracy'), f"{model_data['train_score']:.2%}")
    st.metric(t('testing_accuracy'), f"{model_data['test_score']:.2%}")
    st.metric(t('dataset_size'), f"{len(df):,} {t('samples')}")
    
    if model_data.get('xgb_score') is not None:
        st.markdown("---")
        st.markdown(f"### {t('model_comparison')}")
        st.write(f"**{t('random_forest')}:** {model_data['rf_score']:.2%}")
        st.write(f"**{t('xgboost')}:** {model_data['xgb_score']:.2%}")
        improvement = (model_data['xgb_score'] - model_data['rf_score']) * 100
        if improvement > 0:
            st.success(f"✅ {t('xgb_improved')} {improvement:.1f}%")
        elif improvement < 0:
            st.info(f"ℹ️ {t('rf_better')} {abs(improvement):.1f}%")
        else:
            st.info(t('models_equal'))
    
    st.markdown("---")
    st.markdown(f"### {t('dataset_stats')}")
    st.write(f"**{t('age_range')}:** {df['age'].min()} - {df['age'].max()} {t('years')}")
    st.write(f"**{t('bmi_range')}:** {df['bmi'].min():.1f} - {df['bmi'].max():.1f}")
    st.write(f"**{t('avg_cost')}:** ₹{df['charges'].mean():,.2f}")
    st.write(f"**{t('smokers')}:** {(df['smoker'] == 'yes').sum()} ({(df['smoker'] == 'yes').mean()*100:.1f}%)")
    
    st.markdown("---")
    st.markdown(f"### {t('export_data')}")
    st.metric(t('predictions_made'), len(st.session_state.prediction_history))
    
    if len(st.session_state.prediction_history) > 0:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        csv_data = history_df.to_csv(index=False)
        
        st.download_button(
            label=t('download_csv'),
            data=csv_data,
            file_name=f"insurance_predictions_history.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        if st.button(t('clear_history'), use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info(t('no_predictions'))

# Main content tabs
admin_email = "kit27.ad17@gmail.com"
show_admin = st.session_state.email == admin_email

tab_list = [
    t('tab_prediction'), 
    t('tab_visualizations'), 
    t('tab_whatif'), 
    t('tab_cost_comparison'), 
    t('tab_accident'),
    t('tab_trends'),
    t('tab_chatbot'),
    t('tab_document'),
    t('tab_quotes'),
    t('tab_tax'),
    t('tab_receipt_analyzer')
]

if show_admin:
    tab_list.append(t('tab_admin'))

tabs = st.tabs(tab_list)

# Assign tabs to variables
if show_admin:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = tabs
else:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = tabs
    tab12 = None # Admin tab hidden

# Tab 1: Prediction
with tab1:
    st.header(t('insurance_cost_prediction'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t('personal_info'))
        age = st.slider(t('age'), min_value=18, max_value=64, value=30, help=t('age_help'))
        sex = st.selectbox(t('gender'), options=['male', 'female'], format_func=lambda x: t(x))
        children = st.number_input(t('children'), min_value=0, max_value=5, value=0, step=1)
        region = st.selectbox(t('region'), options=['northeast', 'northwest', 'southeast', 'southwest'], format_func=lambda x: t(x))
    
    with col2:
        st.subheader(t('health_info'))
        bmi = st.slider(t('bmi'), min_value=15.0, max_value=50.0, value=25.0, step=0.1,
                       help=t('bmi_help'))
        smoker = st.selectbox(t('smoking_status'), options=['no', 'yes'], format_func=lambda x: t(x))
        
        # BMI category display
        if bmi < 18.5:
            bmi_category = t('underweight')
        elif bmi < 25:
            bmi_category = t('normal_weight')
        elif bmi < 30:
            bmi_category = t('overweight')
        else:
            bmi_category = t('obese')
        st.info(f"{t('bmi_category')}: **{bmi_category}**")
    
    # Predict button
    if st.button(t('predict_button'), type="primary", use_container_width=True):
        # Make prediction
        predicted_cost = predict_cost(model_data, age, sex, bmi, children, smoker, region)
        risk_level, risk_icon = get_risk_level(predicted_cost)
        
        # Save to prediction history
        from datetime import datetime
        prediction_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region,
            'predicted_cost': predicted_cost,
            'risk_level': risk_level,
            'monthly_premium': predicted_cost / 12
        }
        st.session_state.prediction_history.append(prediction_record)
        
        # Display results
        st.markdown("---")
        st.subheader(t('prediction_results'))
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric(t('predicted_annual_cost'), f"₹{predicted_cost:,.2f}")
        
        with result_col2:
            st.metric(t('risk_level'), f"{risk_icon} {risk_level}")
        
        with result_col3:
            monthly_cost = predicted_cost / 12
            st.metric(t('monthly_premium'), f"₹{monthly_cost:,.2f}")
        
        # Cost breakdown
        st.markdown("---")
        st.subheader(t('cost_factor_analysis'))
        
        # Calculate impact of each factor
        base_prediction = predict_cost(model_data, 30, 'male', 25, 0, 'no', 'northeast')
        
        factor_impacts = {
            t('age_factor'): ((age - 30) * 250),
            t('bmi_factor'): ((bmi - 25) * 200) if bmi > 25 else 0,
            t('smoking_factor'): (predicted_cost * 0.6) if smoker == 'yes' else 0,
            t('children_factor'): children * 500,
        }
        
        impact_df = pd.DataFrame({
            'Factor': factor_impacts.keys(),
            'Impact (₹)': factor_impacts.values()
        })
        
        fig_impact = px.bar(impact_df, x='Factor', y='Impact (₹)', 
                           title=t('factor_impact_title'),
                           color='Impact (₹)',
                           color_continuous_scale='Viridis',
                           template='plotly_dark')
        fig_impact.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family='Outfit'
        )
        st.plotly_chart(fig_impact, use_container_width=True)
        
        # PDF Export
        st.markdown("---")
        st.subheader(t('export_report'))
        
        user_data = {
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }
        comparison_data = get_govt_vs_private_comparison(predicted_cost)
        
        pdf_buffer = generate_pdf_report(user_data, predicted_cost, risk_level, comparison_data, factor_impacts)
        
        st.download_button(
            label=t('download_pdf'),
            data=pdf_buffer,
            file_name=f"insurance_prediction_report_{age}y_{sex}_{region}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# Tab 2: Visualizations
with tab2:
    st.header(t('interactive_visualizations'))
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Cost vs Age
        fig_age = px.scatter(df, x='age', y='charges', color='smoker',
                            title=t('cost_vs_age'),
                            labels={'charges': t('insurance_cost'), 'age': t('age_years')},
                            color_discrete_map={'yes': '#f43f5e', 'no': '#10b981'},
                            trendline='lowess',
                            template='plotly_dark')
        fig_age.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family='Outfit'
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Cost vs Children
        avg_by_children = df.groupby('children')['charges'].mean().reset_index()
        fig_children = px.bar(avg_by_children, x='children', y='charges',
                             title=t('avg_cost_children'),
                             labels={'charges': t('average_cost'), 'children': t('number_of_children')},
                             template='plotly_dark',
                             color_discrete_sequence=['#6366f1'])
        fig_children.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family='Outfit'
        )
        st.plotly_chart(fig_children, use_container_width=True)
    
    with viz_col2:
        # Cost vs BMI
        fig_bmi = px.scatter(df, x='bmi', y='charges', color='smoker',
                            title=t('cost_vs_bmi'),
                            labels={'charges': t('insurance_cost'), 'bmi': 'BMI'},
                            color_discrete_map={'yes': '#f43f5e', 'no': '#10b981'},
                            trendline='lowess',
                            template='plotly_dark')
        fig_bmi.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family='Outfit'
        )
        st.plotly_chart(fig_bmi, use_container_width=True)
        
        # Smoking Impact
        avg_by_smoker = df.groupby('smoker')['charges'].mean().reset_index()
        fig_smoker = px.bar(avg_by_smoker, x='smoker', y='charges',
                           title=t('smoking_impact'),
                           labels={'charges': t('average_cost'), 'smoker': t('smoker')},
                           color='smoker',
                           color_discrete_map={'yes': '#f43f5e', 'no': '#10b981'},
                           template='plotly_dark')
        fig_smoker.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family='Outfit'
        )
        st.plotly_chart(fig_smoker, use_container_width=True)
    
    # Regional analysis
    st.markdown("---")
    regional_stats = df.groupby('region')['charges'].agg(['mean', 'min', 'max']).reset_index()
    regional_stats.columns = [t('region'), t('average') + ' Cost', t('minimum') + ' Cost', t('maximum') + ' Cost']
    fig_region = go.Figure()
    fig_region.add_trace(go.Bar(name=t('average'), x=regional_stats[t('region')], y=regional_stats[t('average') + ' Cost'], marker_color='#6366f1'))
    fig_region.add_trace(go.Bar(name=t('minimum'), x=regional_stats[t('region')], y=regional_stats[t('minimum') + ' Cost'], marker_color='#10b981'))
    fig_region.add_trace(go.Bar(name=t('maximum'), x=regional_stats[t('region')], y=regional_stats[t('maximum') + ' Cost'], marker_color='#f43f5e'))
    fig_region.update_layout(
        title=t('regional_cost_analysis'), 
        barmode='group', 
        height=400,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family='Outfit'
    )
    st.plotly_chart(fig_region, use_container_width=True)

# Tab 3: What-If Analysis
with tab3:
    st.header(t('whatif_tool'))
    st.markdown(t('whatif_description'))
    
    # Store baseline values in session state
    if 'baseline_age' not in st.session_state:
        st.session_state.baseline_age = 30
        st.session_state.baseline_sex = 'male'
        st.session_state.baseline_bmi = 25.0
        st.session_state.baseline_children = 0
        st.session_state.baseline_smoker = 'no'
        st.session_state.baseline_region = 'northeast'
    
    baseline_col, whatif_col = st.columns(2)
    
    with baseline_col:
        st.subheader(t('baseline_scenario'))
        base_age = st.slider(t('baseline_age'), 18, 64, st.session_state.baseline_age, key='base_age')
        base_sex = st.selectbox(t('baseline_gender'), ['male', 'female'], 
                               index=0 if st.session_state.baseline_sex == 'male' else 1, key='base_sex', format_func=lambda x: t(x))
        base_bmi = st.slider(t('baseline_bmi'), 15.0, 50.0, st.session_state.baseline_bmi, 0.1, key='base_bmi')
        base_children = st.number_input(t('baseline_children'), 0, 5, st.session_state.baseline_children, key='base_children')
        base_smoker = st.selectbox(t('baseline_smoker'), ['no', 'yes'],
                                  index=0 if st.session_state.baseline_smoker == 'no' else 1, key='base_smoker', format_func=lambda x: t(x))
        base_region = st.selectbox(t('baseline_region'), ['northeast', 'northwest', 'southeast', 'southwest'],
                                  index=['northeast', 'northwest', 'southeast', 'southwest'].index(st.session_state.baseline_region),
                                  key='base_region', format_func=lambda x: t(x))
        
        baseline_cost = predict_cost(model_data, base_age, base_sex, base_bmi, base_children, base_smoker, base_region)
        st.metric(t('baseline_cost'), f"₹{baseline_cost:,.2f}")
    
    with whatif_col:
        st.subheader(t('whatif_scenario'))
        whatif_age = st.slider(t('whatif_age'), 18, 64, base_age, key='whatif_age')
        whatif_sex = st.selectbox(t('whatif_gender'), ['male', 'female'], 
                                 index=0 if base_sex == 'male' else 1, key='whatif_sex', format_func=lambda x: t(x))
        whatif_bmi = st.slider(t('whatif_bmi'), 15.0, 50.0, base_bmi, 0.1, key='whatif_bmi')
        whatif_children = st.number_input(t('whatif_children'), 0, 5, base_children, key='whatif_children')
        whatif_smoker = st.selectbox(t('whatif_smoker'), ['no', 'yes'],
                                    index=0 if base_smoker == 'no' else 1, key='whatif_smoker', format_func=lambda x: t(x))
        whatif_region = st.selectbox(t('whatif_region'), ['northeast', 'northwest', 'southeast', 'southwest'],
                                    index=['northeast', 'northwest', 'southeast', 'southwest'].index(base_region),
                                    key='whatif_region', format_func=lambda x: t(x))
        
        whatif_cost = predict_cost(model_data, whatif_age, whatif_sex, whatif_bmi, whatif_children, whatif_smoker, whatif_region)
        cost_difference = whatif_cost - baseline_cost
        percent_change = (cost_difference / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        st.metric(t('whatif_cost'), f"₹{whatif_cost:,.2f}", 
                 delta=f"₹{cost_difference:,.2f} ({percent_change:+.1f}%)")
    
    # Comparison visualization
    st.markdown("---")
    st.subheader(t('scenario_comparison'))
    
    comparison_data = pd.DataFrame({
        'Scenario': [t('baseline'), t('whatif')],
        'Cost': [baseline_cost, whatif_cost]
    })
    
    fig_comparison = px.bar(comparison_data, x='Scenario', y='Cost',
                           title=t('comparison_title'),
                           color='Scenario',
                           color_discrete_map={t('baseline'): '#6366f1', t('whatif'): '#f43f5e'},
                           template='plotly_dark')
    fig_comparison.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_family='Outfit'
    )
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Parameter change summary
    st.subheader(t('parameter_changes'))
    changes = []
    if base_age != whatif_age:
        changes.append(f"{t('age')}: {base_age} → {whatif_age}")
    if base_sex != whatif_sex:
        changes.append(f"{t('gender')}: {t(base_sex)} → {t(whatif_sex)}")
    if base_bmi != whatif_bmi:
        changes.append(f"BMI: {base_bmi:.1f} → {whatif_bmi:.1f}")
    if base_children != whatif_children:
        changes.append(f"{t('children')}: {base_children} → {whatif_children}")
    if base_smoker != whatif_smoker:
        changes.append(f"{t('smoker')}: {t(base_smoker)} → {t(whatif_smoker)}")
    if base_region != whatif_region:
        changes.append(f"{t('region')}: {t(base_region)} → {t(whatif_region)}")
    
    if changes:
        for change in changes:
            st.write(f"• {change}")
    else:
        st.info(t('no_changes'))

# Tab 4: Cost Comparison
with tab4:
    st.header(t('govt_vs_private'))
    st.markdown(t('govt_vs_private_desc'))
    
    # Insurance Company Data
    insurance_companies = {
        'Life Insurance Corporation of India (LIC)': {
            'type': 'Life Insurance',
            'life': True,
            'general': False,
            'health': False
        },
        'HDFC Life': {
            'type': 'Life Insurance',
            'life': True,
            'general': False,
            'health': False
        },
        'ICICI Prudential Life': {
            'type': 'Life Insurance',
            'life': True,
            'general': False,
            'health': False
        },
        'SBI Life': {
            'type': 'Life Insurance',
            'life': True,
            'general': False,
            'health': False
        },
        'Max Life': {
            'type': 'Life Insurance',
            'life': True,
            'general': False,
            'health': False
        },
        'Aditya Birla Sun Life': {
            'type': 'Life Insurance',
            'life': True,
            'general': False,
            'health': False
        },
        'Kotak Mahindra Life': {
            'type': 'Life Insurance',
            'life': True,
            'general': False,
            'health': False
        },
        'TATA AIA Life': {
            'type': 'Life Insurance',
            'life': True,
            'general': False,
            'health': False
        },
        'Bajaj Allianz Life': {
            'type': 'Life & General Insurance',
            'life': True,
            'general': True,
            'health': True
        },
        'ICICI Lombard General Insurance': {
            'type': 'General & Health Insurance',
            'life': False,
            'general': True,
            'health': True
        },
        'Star Health & Allied Insurance': {
            'type': 'Stand-alone Health Insurance',
            'life': False,
            'general': True,
            'health': True
        },
        'Aditya Birla Health Insurance': {
            'type': 'Stand-alone Health Insurance',
            'life': False,
            'general': True,
            'health': True
        },
        'Niva Bupa Health Insurance': {
            'type': 'Stand-alone Health Insurance',
            'life': False,
            'general': True,
            'health': True
        },
        'Care Health Insurance': {
            'type': 'Stand-alone Health Insurance',
            'life': False,
            'general': True,
            'health': True
        },
        'Manipal Cigna Health Insurance': {
            'type': 'Stand-alone Health Insurance',
            'life': False,
            'general': True,
            'health': True
        }
    }
    
    # Insurance Company Selector
    st.subheader("🏢 Select Insurance Company")
    
    insurance_filter = st.radio(
        "Filter by Insurance Type",
        ['All Companies', 'Life Insurance', 'General Insurance', 'Health Insurance'],
        horizontal=True
    )
    
    filtered_companies = []
    if insurance_filter == 'All Companies':
        filtered_companies = list(insurance_companies.keys())
    elif insurance_filter == 'Life Insurance':
        filtered_companies = [k for k, v in insurance_companies.items() if v['life']]
    elif insurance_filter == 'General Insurance':
        filtered_companies = [k for k, v in insurance_companies.items() if v['general']]
    elif insurance_filter == 'Health Insurance':
        filtered_companies = [k for k, v in insurance_companies.items() if v['health']]
    
    selected_company = st.selectbox(
        "Choose Insurance Company",
        filtered_companies,
        index=0 if filtered_companies else None
    )
    
    if selected_company:
        company_info = insurance_companies[selected_company]
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            life_icon = "✅" if company_info['life'] else "❌"
            st.metric("Life Insurance", life_icon)
        
        with col_info2:
            general_icon = "✅" if company_info['general'] else "❌"
            st.metric("General Insurance", general_icon)
        
        with col_info3:
            health_icon = "✅" if company_info['health'] else "❌"
            st.metric("Health Insurance", health_icon)
        
        st.info(f"📋 **Company Type:** {company_info['type']}")
    
    st.markdown("---")
    
    # Input section
    st.subheader(t('enter_details'))
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        comp_age = st.slider(t('age'), 18, 64, 35, key='comp_age')
        comp_sex = st.selectbox(t('gender'), ['male', 'female'], key='comp_sex', format_func=lambda x: t(x))
        comp_bmi = st.slider(t('bmi'), 15.0, 50.0, 27.0, 0.1, key='comp_bmi')
    
    with comp_col2:
        comp_children = st.number_input(t('children'), 0, 5, 1, key='comp_children')
        comp_smoker = st.selectbox(t('smoker'), ['no', 'yes'], key='comp_smoker', format_func=lambda x: t(x))
        comp_region = st.selectbox(t('region'), ['northeast', 'northwest', 'southeast', 'southwest'], key='comp_region', format_func=lambda x: t(x))
    
    if st.button(t('compare_button'), type="primary", use_container_width=True):
        # Predict cost
        predicted_cost = predict_cost(model_data, comp_age, comp_sex, comp_bmi, comp_children, comp_smoker, comp_region)
        comparison = get_govt_vs_private_comparison(predicted_cost)
        risk_level_comp, _ = get_risk_level(predicted_cost)
        
        # Save to prediction history
        from datetime import datetime
        prediction_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'age': comp_age,
            'sex': comp_sex,
            'bmi': comp_bmi,
            'children': comp_children,
            'smoker': comp_smoker,
            'region': comp_region,
            'predicted_cost': predicted_cost,
            'risk_level': risk_level_comp,
            'monthly_premium': predicted_cost / 12
        }
        st.session_state.prediction_history.append(prediction_record)
        
        st.markdown("---")
        st.subheader("Cost Comparison Results")
        
        # Display selected company information and adjust costs
        health_multiplier = 1.0
        if selected_company:
            st.success(f"✅ **Selected Insurance Provider:** {selected_company}")
            company_info = insurance_companies[selected_company]
            
            provider_col1, provider_col2 = st.columns(2)
            with provider_col1:
                st.markdown(f"**Provider Type:** {company_info['type']}")
            
            with provider_col2:
                coverage_types = []
                if company_info['health']:
                    coverage_types.append("Health")
                if company_info['life']:
                    coverage_types.append("Life")
                if company_info['general']:
                    coverage_types.append("General")
                st.markdown(f"**Available Coverage:** {', '.join(coverage_types)}")
            
            if company_info['health']:
                if 'Stand-alone' in company_info['type']:
                    health_multiplier = 0.95
                    st.info("💡 This stand-alone health insurer typically offers specialized health coverage with competitive premiums (5% discount applied)")
                elif company_info['life'] and company_info['health']:
                    health_multiplier = 1.05
                    st.info("💡 This multi-type insurer offers bundled benefits but may have slightly higher premiums (5% markup)")
                else:
                    st.info("💡 Standard health insurance rates apply for this provider")
            else:
                st.warning("⚠️ This company doesn't offer health insurance. Consider selecting a health insurance provider for medical coverage.")
        
        comparison['private_base'] = comparison['private_base'] * health_multiplier
        comparison['private_premium'] = comparison['private_premium'] * health_multiplier
        
        st.markdown("---")
        
        # Create comparison cards
        govt_col, private_col = st.columns(2)
        
        with govt_col:
            st.markdown("### 🏛️ Government Scheme")
            st.metric("Government Coverage", f"₹{comparison['govt_coverage']:,.2f}")
            st.metric("Your Out-of-Pocket", f"₹{comparison['govt_out_of_pocket']:,.2f}")
            st.metric("Coverage Percentage", f"{(comparison['govt_coverage']/predicted_cost)*100:.1f}%")
            
            st.markdown("**Pros:**")
            st.write("• Lower premiums")
            st.write("• Basic coverage included")
            st.write("• Government subsidized")
            
            st.markdown("**Cons:**")
            st.write("• Limited coverage")
            st.write("• Higher out-of-pocket costs")
            st.write("• Fewer hospital choices")
        
        with private_col:
            st.markdown("### 🏥 Private Insurance")
            st.metric("Base Plan Cost", f"₹{comparison['private_base']:,.2f}")
            st.metric("Premium Plan Cost", f"₹{comparison['private_premium']:,.2f}")
            avg_private = (comparison['private_base'] + comparison['private_premium']) / 2
            st.metric("Coverage Percentage", f"{(avg_private/predicted_cost)*100:.1f}%")
            
            st.markdown("**Pros:**")
            st.write("• Comprehensive coverage")
            st.write("• Wide hospital network")
            st.write("• Additional benefits")
            
            st.markdown("**Cons:**")
            st.write("• Higher premiums")
            st.write("• Complex terms")
            st.write("• Waiting periods")
        
        # Visual comparison
        st.markdown("---")
        st.subheader("Visual Cost Breakdown")
        
        comparison_df = pd.DataFrame({
            'Insurance Type': ['Government\nCoverage', 'Government\nOut-of-Pocket', 
                             'Private\nBase Plan', 'Private\nPremium Plan'],
            'Cost (₹)': [comparison['govt_coverage'], comparison['govt_out_of_pocket'],
                        comparison['private_base'], comparison['private_premium']],
            'Category': ['Government', 'Government', 'Private', 'Private']
        })
        
        fig_comp = px.bar(comparison_df, x='Insurance Type', y='Cost (₹)', 
                         color='Category',
                         title='Insurance Cost Comparison',
                         color_discrete_map={'Government': '#2ecc71', 'Private': '#3498db'})
        fig_comp.add_hline(y=predicted_cost, line_dash="dash", line_color="red",
                          annotation_text=f"Predicted Total Cost: ₹{predicted_cost:,.2f}")
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if comparison['govt_out_of_pocket'] < comparison['private_base']:
            st.success("✅ Government scheme may be more economical if you can manage the out-of-pocket costs.")
        else:
            st.info("ℹ️ Private insurance might offer better value with comprehensive coverage.")
        
        if comp_smoker == 'yes':
            st.warning("⚠️ As a smoker, consider quitting to significantly reduce insurance costs. Smoking can increase costs by 150-250%.")
        
        if comp_bmi > 30:
            st.warning("⚠️ High BMI increases insurance costs. Consider a weight management program to reduce premiums.")
        
        # Government Scheme Recommendations
        st.markdown("---")
        st.subheader("🏛️ Eligible Government Healthcare Schemes")
        st.markdown("Based on your profile, you may be eligible for the following government assistance programs:")
        
        recommendations = get_government_scheme_recommendations(
            comp_age, comp_children, comp_smoker, predicted_cost, comp_bmi, comp_region
        )
        
        for i, rec in enumerate(recommendations):
            with st.expander(f"{'🔴' if rec['priority'] == 'High' else '🟡'} {rec['name']} - {rec['priority']} Priority"):
                st.markdown(f"**Eligibility:** {rec['eligibility']}")
                st.markdown(f"**Coverage:** {rec['coverage']}")
                
                st.markdown("**Benefits:**")
                for benefit in rec['benefits']:
                    st.write(f"• {benefit}")
                
                st.markdown(f"**How to Apply:** {rec['application']}")
                
                if rec['priority'] == 'High':
                    st.success("✅ This program is highly recommended for your profile")
        
        if len(recommendations) > 0:
            st.info(f"💡 You qualify for {len(recommendations)} government healthcare programs. Consider applying to maximize your coverage and reduce out-of-pocket costs.")

# Tab 5: Accident/Injury Cost Estimation
with tab5:
    st.header("Accident/Injury Cost Estimation")
    st.markdown("""
    Estimate additional insurance costs for accidents or injuries. This helps you understand potential 
    out-of-pocket expenses and plan for unexpected medical events.
    """)
    
    # Personal info for context
    st.subheader("Your Profile")
    acc_col1, acc_col2 = st.columns(2)
    
    with acc_col1:
        acc_age = st.slider("Age", 18, 64, 35, key='acc_age')
        acc_sex = st.selectbox("Gender", ['male', 'female'], key='acc_sex')
        acc_bmi = st.slider("BMI", 15.0, 50.0, 27.0, 0.1, key='acc_bmi')
    
    with acc_col2:
        acc_children = st.number_input("Children", 0, 5, 1, key='acc_children')
        acc_smoker = st.selectbox("Smoker", ['no', 'yes'], key='acc_smoker')
        acc_region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'], key='acc_region')
    
    # Accident/Injury Details
    st.markdown("---")
    st.subheader("Accident/Injury Details")
    
    accident_col1, accident_col2 = st.columns(2)
    
    with accident_col1:
        accident_type = st.selectbox(
            "Type of Accident/Injury",
            ['car accident', 'fall', 'sports injury', 'workplace injury', 'other'],
            help="Select the type of accident or injury"
        )
        
        severity = st.selectbox(
            "Severity Level",
            ['minor', 'moderate', 'severe', 'critical'],
            help="Minor: cuts, bruises | Moderate: sprains, minor fractures | Severe: major fractures, internal injuries | Critical: life-threatening"
        )
        
        recovery_days = st.slider(
            "Estimated Recovery Time (days)",
            min_value=1,
            max_value=365,
            value=30,
            help="Number of days needed for full recovery"
        )
    
    with accident_col2:
        hospitalization = st.selectbox(
            "Hospitalization Required?",
            ['no', 'yes'],
            help="Will you need to stay in the hospital?"
        )
        
        surgery = st.selectbox(
            "Surgery Required?",
            ['no', 'yes'],
            help="Will surgical intervention be necessary?"
        )
        
        st.metric("Recovery Period", f"{recovery_days} days" if recovery_days < 30 else f"{recovery_days//30} months")
    
    # Calculate button
    if st.button("💉 Estimate Accident/Injury Cost", type="primary", use_container_width=True):
        # Get base insurance cost
        base_cost = predict_cost(model_data, acc_age, acc_sex, acc_bmi, acc_children, acc_smoker, acc_region)
        
        # Get accident/injury cost
        accident_cost = estimate_accident_injury_cost(accident_type, severity, hospitalization, surgery, recovery_days)
        
        # Total cost
        total_cost = base_cost + accident_cost
        
        st.markdown("---")
        st.subheader("Cost Estimation Results")
        
        # Display metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Base Annual Insurance", f"₹{base_cost:,.2f}")
        
        with metric_col2:
            st.metric("Accident/Injury Cost", f"₹{accident_cost:,.2f}")
        
        with metric_col3:
            st.metric("Total Cost", f"₹{total_cost:,.2f}")
        
        with metric_col4:
            increase_pct = (accident_cost / base_cost) * 100
            st.metric("Cost Increase", f"{increase_pct:.0f}%")
        
        # Cost breakdown
        st.markdown("---")
        st.subheader("Cost Breakdown")
        
        breakdown = get_accident_cost_breakdown(accident_type, severity, hospitalization, surgery, recovery_days)
        
        breakdown_df = pd.DataFrame({
            'Component': list(breakdown.keys()),
            'Cost (₹)': list(breakdown.values())
        })
        
        fig_breakdown = px.bar(breakdown_df, x='Component', y='Cost (₹)',
                              title='Detailed Cost Breakdown',
                              color='Cost (₹)',
                              color_continuous_scale='Reds')
        fig_breakdown.update_layout(height=400)
        st.plotly_chart(fig_breakdown, use_container_width=True)
        
        # Financial Planning
        st.markdown("---")
        st.subheader("💰 Financial Planning")
        
        plan_col1, plan_col2 = st.columns(2)
        
        with plan_col1:
            st.markdown("### Immediate Costs")
            st.write(f"**Emergency Treatment:** ₹{breakdown.get('Base Treatment Cost', 0):,.2f}")
            if 'Hospitalization' in breakdown:
                st.write(f"**Hospital Stay:** ₹{breakdown['Hospitalization']:,.2f}")
            if 'Surgery' in breakdown:
                st.write(f"**Surgery:** ₹{breakdown['Surgery']:,.2f}")
        
        with plan_col2:
            st.markdown("### Ongoing Costs")
            if 'Daily Hospital Care' in breakdown:
                st.write(f"**Daily Care ({recovery_days} days):** ₹{breakdown['Daily Hospital Care']:,.2f}")
            st.write(f"**Recovery & Medication:** ₹{breakdown.get('Recovery & Medication', 0):,.2f}")
            st.write(f"**Monthly Average:** ₹{accident_cost/12:,.2f}")
        
        # Insurance Coverage Estimates
        st.markdown("---")
        st.subheader("📋 Insurance Coverage Estimates")
        
        govt_accident_coverage = min(accident_cost * 0.5, 10000)
        private_accident_coverage = accident_cost * 0.85
        
        coverage_col1, coverage_col2 = st.columns(2)
        
        with coverage_col1:
            st.markdown("### 🏛️ Government Insurance")
            st.metric("Estimated Coverage", f"₹{govt_accident_coverage:,.2f}")
            st.metric("Your Out-of-Pocket", f"₹{accident_cost - govt_accident_coverage:,.2f}")
            coverage_pct = (govt_accident_coverage / accident_cost) * 100
            st.metric("Coverage %", f"{coverage_pct:.1f}%")
        
        with coverage_col2:
            st.markdown("### 🏥 Private Insurance")
            st.metric("Estimated Coverage", f"₹{private_accident_coverage:,.2f}")
            st.metric("Your Out-of-Pocket", f"₹{accident_cost - private_accident_coverage:,.2f}")
            coverage_pct = (private_accident_coverage / accident_cost) * 100
            st.metric("Coverage %", f"{coverage_pct:.1f}%")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if severity in ['severe', 'critical']:
            st.error("⚠️ This is a serious medical event. Ensure you have comprehensive insurance coverage.")
        
        if accident_cost > 20000:
            st.warning("💰 High medical costs detected. Consider premium insurance plans for better protection.")
        
        if hospitalization == 'yes' and surgery == 'yes':
            st.info("🏥 Complex treatment requires both hospitalization and surgery. Private insurance may provide better coverage.")
        
        if accident_type == 'car accident':
            st.info("🚗 Car accident victims may be eligible for additional compensation through auto insurance claims.")
        
        if accident_type == 'workplace injury':
            st.info("👷 Workplace injuries may be covered under worker's compensation. Check with your employer.")

# Tab 6: Cost Trends Dashboard
with tab6:
    st.header(t('cost_trends_dashboard'))
    st.markdown(t('trends_description'))
    
    if len(st.session_state.prediction_history) > 0:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(t('predictions_count'), len(history_df))
        with col2:
            st.metric(t('avg_predicted_cost'), f"₹{history_df['predicted_cost'].mean():,.2f}")
        with col3:
            cost_range = history_df['predicted_cost'].max() - history_df['predicted_cost'].min()
            st.metric(t('cost_range'), f"₹{cost_range:,.2f}")
        
        # Trend over time
        st.markdown("---")
        st.subheader(t('trend_over_time'))
        fig_trend = px.line(history_df, x='timestamp', y='predicted_cost',
                           title=t('trend_over_time'),
                           labels={'predicted_cost': t('insurance_cost'), 'timestamp': 'Time'},
                           template='plotly_dark')
        fig_trend.update_traces(mode='lines+markers', line_color='#6366f1', marker=dict(size=8, color='#2dd4bf'))
        fig_trend.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_family='Outfit'
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Additional analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(t('cost_by_age_group'))
            history_df['age_group'] = pd.cut(history_df['age'], bins=[0, 30, 40, 50, 65], labels=['18-30', '31-40', '41-50', '51-64'])
            age_group_avg = history_df.groupby('age_group')['predicted_cost'].mean().reset_index()
            fig_age = px.bar(age_group_avg, x='age_group', y='predicted_cost',
                           labels={'predicted_cost': t('average_cost'), 'age_group': t('age')},
                           template='plotly_dark',
                           color_discrete_sequence=['#2dd4bf'])
            fig_age.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family='Outfit'
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.subheader(t('cost_by_smoker'))
            smoker_dist = history_df.groupby('smoker')['predicted_cost'].mean().reset_index()
            fig_smoker = px.pie(smoker_dist, values='predicted_cost', names='smoker',
                               title=t('cost_by_smoker'),
                               template='plotly_dark',
                               color_discrete_sequence=['#6366f1', '#f43f5e'])
            fig_smoker.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family='Outfit'
            )
            st.plotly_chart(fig_smoker, use_container_width=True)
        
        # Highest and lowest predictions
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            max_idx = history_df['predicted_cost'].idxmax()
            max_pred = history_df.loc[max_idx]
            st.metric(t('highest_cost'), f"₹{max_pred['predicted_cost']:,.2f}", 
                     f"Age: {max_pred['age']}, Smoker: {max_pred['smoker']}")
        with col2:
            min_idx = history_df['predicted_cost'].idxmin()
            min_pred = history_df.loc[min_idx]
            st.metric(t('lowest_cost'), f"₹{min_pred['predicted_cost']:,.2f}",
                     f"Age: {min_pred['age']}, Smoker: {min_pred['smoker']}")
    else:
        st.info(t('no_trends_data'))

# Tab 7: AI Chatbot
with tab7:
    st.header(t('ai_chatbot'))
    st.markdown(t('chatbot_description'))
    
    # Groq API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    import requests

    if not GROQ_API_KEY:
        st.warning("Groq API Key (GROQ_API_KEY) is missing in .env file. Please provide a valid key.")
    else:
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input(t('chatbot_placeholder')):
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner(t('chatbot_thinking')):
                    try:
                        # Context about insurance
                        system_prompt = """You are an expert health insurance advisor in India. 
                        Help users understand health insurance concepts, coverage options, premiums, 
                        tax benefits under Section 80D, and provide personalized advice based on their needs.
                        Be concise, accurate, and helpful. Use simple language."""
                        
                        payload = {
                            "model": "llama-3.3-70b-versatile",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]
                            ],
                            "temperature": 0.7,
                            "max_tokens": 1024
                        }
                        
                        headers = {
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        }
                        
                        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
                        
                        if response.status_code == 200:
                            result = response.json()
                            ai_response = result['choices'][0]['message']['content']
                            st.markdown(ai_response)
                            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                        else:
                            st.error(f"API Error ({response.status_code}): {response.text}")
                    except Exception as e:
                        if "timeout" in str(e).lower():
                            st.error("Request timeout. The Groq service took too long to respond. Please try again.")
                        else:
                            st.error(f"Chatbot Error: {str(e)}")

# Tab 8: Document Analyzer
with tab8:
    st.header(t('document_analyzer'))
    st.markdown(t('doc_description'))
    
    uploaded_file = st.file_uploader(t('upload_policy'), type=['pdf'])
    
    if uploaded_file is not None:
        # Validate file size (max 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("File size too large. Please upload a PDF smaller than 10MB.")
        elif st.button(t('analyze_button'), type="primary"):
            with st.spinner(t('analyzing')):
                try:
                    # Read PDF content
                    from PyPDF2 import PdfReader
                    reader = PdfReader(uploaded_file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    
                    # Display analysis (simplified version)
                    st.success(t('analysis_results'))
                    
                    # Key points extraction (simplified)
                    st.subheader(t('key_points'))
                    st.markdown(f"""
                    - Document contains {len(reader.pages)} pages
                    - Approximately {len(text.split())} words
                    - Policy document uploaded successfully
                    """)
                    
                    # Show sample text
                    st.subheader(t('coverage_details'))
                    st.text_area("Document Preview", text[:1000] + "...", height=200)
                    
                    # AI analysis if OpenAI is available
                    if 'OPENAI_API_KEY' in os.environ:
                        try:
                            from openai import OpenAI
                            client = OpenAI()
                            
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are an insurance policy analyst. Analyze this policy document and extract key information."},
                                    {"role": "user", "content": f"Analyze this insurance policy and provide: 1) Key coverage details, 2) Exclusions, 3) Premium information. Document text: {text[:3000]}"}
                                ]
                            )
                            
                            st.subheader("AI Analysis")
                            st.markdown(response.choices[0].message.content)
                        except:
                            pass
                    
                except Exception as e:
                    if "encrypted" in str(e).lower():
                        st.error("This PDF appears to be encrypted or password-protected. Please upload an unencrypted PDF.")
                    elif "damaged" in str(e).lower() or "invalid" in str(e).lower():
                        st.error("This PDF appears to be corrupted or invalid. Please upload a valid PDF document.")
                    else:
                        st.error(f"Error analyzing document: {str(e)}")
    else:
        st.info(t('no_document'))

# Tab 9: Real-time Insurance Quotes
with tab9:
    st.header(t('realtime_quotes'))
    st.markdown(t('quotes_description'))
    st.info("ℹ️ These are simulated quotes for demonstration. For actual quotes, please contact insurance providers directly.")
    
    # User profile for quotes
    quote_col1, quote_col2 = st.columns(2)
    
    with quote_col1:
        quote_age = st.slider(t('age'), 18, 64, 30, key='quote_age')
        quote_sex = st.selectbox(t('gender'), ['male', 'female'], key='quote_sex', format_func=lambda x: t(x))
        quote_bmi = st.slider(t('bmi'), 15.0, 50.0, 25.0, 0.1, key='quote_bmi')
    
    with quote_col2:
        quote_children = st.number_input(t('children'), 0, 5, 0, key='quote_children')
        quote_smoker = st.selectbox(t('smoking_status'), ['no', 'yes'], key='quote_smoker', format_func=lambda x: t(x))
        quote_region = st.selectbox(t('region'), ['northeast', 'northwest', 'southeast', 'southwest'], key='quote_region', format_func=lambda x: t(x))
    
    if st.button(t('get_quotes'), type="primary"):
        with st.spinner(t('fetching_quotes')):
            # Calculate base prediction
            base_cost = predict_cost(model_data, quote_age, quote_sex, quote_bmi, quote_children, quote_smoker, quote_region)
            
            # Generate mock quotes from different providers
            providers = [
                {"name": "HDFC ERGO", "multiplier": 0.95, "coverage": "5 Lakh"},
                {"name": "ICICI Lombard", "multiplier": 1.0, "coverage": "5 Lakh"},
                {"name": "Star Health", "multiplier": 0.92, "coverage": "5 Lakh"},
                {"name": "Care Health", "multiplier": 0.98, "coverage": "5 Lakh"},
                {"name": "Max Bupa", "multiplier": 1.05, "coverage": "5 Lakh"},
            ]
            
            st.subheader(t('available_plans'))
            
            for provider in providers:
                premium = base_cost * provider['multiplier']
                with st.expander(f"{provider['name']} - ₹{premium:,.2f}/year"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(t('annual_premium'), f"₹{premium:,.2f}")
                        st.metric(t('coverage_amount'), provider['coverage'])
                    with col2:
                        st.markdown(f"**{t('key_features')}:**")
                        st.markdown("""
                        - Cashless hospitalization
                        - Pre and post hospitalization
                        - Ambulance charges
                        - Day care procedures
                        """)
            
            st.info(t('quotes_disclaimer'))

# Tab 10: Tax Benefit Calculator
with tab10:
    st.header(t('tax_calculator'))
    st.markdown(t('tax_description'))
    
    tax_col1, tax_col2 = st.columns(2)
    
    with tax_col1:
        st.subheader(t('personal_info'))
        self_premium = st.number_input(t('premium_paid'), 0, 100000, 25000, 1000)
        self_age_cat = st.selectbox(t('age_category'), [t('below_60'), t('above_60')])
        
    with tax_col2:
        st.subheader("Parents")
        parents_premium = st.number_input(t('parents_premium'), 0, 100000, 0, 1000)
        parents_age_cat = st.selectbox(t('parents_age'), [t('below_60'), t('above_60')])
    
    checkup_cost = st.number_input(t('preventive_checkup'), 0, 5000, 0, 500)
    
    if st.button(t('calculate_tax'), type="primary"):
        # Calculate deductions
        self_limit = 50000 if t('above_60') in self_age_cat else 25000
        parents_limit = 50000 if t('above_60') in parents_age_cat else 25000
        
        self_deduction = min(self_premium, self_limit)
        parents_deduction = min(parents_premium, parents_limit)
        checkup_deduction = min(checkup_cost, 5000)
        
        # Checkup is included in the limits, not additional
        total_deduction = min(self_deduction + parents_deduction, 100000)
        
        st.markdown("---")
        st.subheader(t('tax_benefit_results'))
        
        # Display deductions
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        
        with result_col1:
            st.metric(t('self_deduction'), f"₹{self_deduction:,}")
        with result_col2:
            st.metric(t('parents_deduction'), f"₹{parents_deduction:,}")
        with result_col3:
            st.metric(t('checkup_deduction'), f"₹{checkup_deduction:,}")
        with result_col4:
            st.metric(t('total_deduction'), f"₹{total_deduction:,}")
        
        # Tax savings
        st.markdown("---")
        st.subheader("Tax Savings by Bracket")
        
        savings_col1, savings_col2, savings_col3 = st.columns(3)
        
        with savings_col1:
            tax_30 = total_deduction * 0.30
            st.metric(t('tax_saved_30'), f"₹{tax_30:,}")
        with savings_col2:
            tax_20 = total_deduction * 0.20
            st.metric(t('tax_saved_20'), f"₹{tax_20:,}")
        with savings_col3:
            tax_10 = total_deduction * 0.10
            st.metric(t('tax_saved_10'), f"₹{tax_10:,}")
        
        # Information
        st.markdown("---")
        st.info(f"""
        **{t('section_80d_info')}**
        
        {t('deduction_limits')}:
        """)

# Tab 11: Medical Receipt Analyzer
with tab11:
    st.header(t('receipt_analyzer_title'))
    st.markdown(t('receipt_analyzer_desc'))
    
    # Groq Configuration for Vision/OCR
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    uploaded_file = st.file_uploader(t('upload_receipt'), type=['pdf', 'png', 'jpg', 'jpeg', 'webp'])
    
    if uploaded_file is not None:
        file_type = uploaded_file.type
        
        # Display preview
        if 'image' in file_type:
            st.image(uploaded_file, caption=t('upload_receipt'), use_container_width=True)
        
        if st.button(t('analyze_receipt_button'), type="primary", use_container_width=True):
            if not GROQ_API_KEY:
                st.error("Groq API Key (GROQ_API_KEY) is missing in .env file.")
            else:
                with st.spinner(t('analyzing')):
                    try:
                        analysis_text = ""
                        model = "meta-llama/llama-4-scout-17b-16e-instruct" # vision-capable replacement
                        
                        if 'pdf' in file_type:
                            try:
                                import PyPDF2
                                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                                text = ""
                                for page in pdf_reader.pages:
                                    text += page.extract_text()
                                if not text.strip():
                                    st.warning("Could not extract text from PDF. Attempting to analyze layout...")
                                    text = "PDF document (layout analysis needed)"
                                
                                analysis_text = f"Analyze the following medical receipt/prescription text and extract medicine names, dosages, doctor's instructions, and key medical details:\n\n{text}"
                                model = "llama-3.1-8b-instant"
                            except ImportError:
                                st.error("PyPDF2 library not found. Please install it with 'pip install PyPDF2'.")
                                st.stop()
                        else:
                            # Image processing with Vision API
                            base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
                            analysis_text = "Analyze this medical receipt or doctor's prescription image. Extract: 1. Medicine names and dosages 2. Doctor's instructions 3. Key medical details (diagnosis, symptoms if mentioned). Be concise and accurate."
                            model = "meta-llama/llama-4-scout-17b-16e-instruct"
                        
                        # Prepare API call
                        if 'image' in file_type:
                            payload = {
                                "model": model,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": analysis_text},
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:{file_type};base64,{base64_image}"
                                                }
                                            }
                                        ]
                                    }
                                ],
                                "temperature": 0.5,
                                "max_tokens": 1024
                            }
                        else:
                            payload = {
                                "model": model,
                                "messages": [
                                    {"role": "system", "content": "You are a medical document analyzer. Extract medicines and instructions accurately."},
                                    {"role": "user", "content": analysis_text}
                                ],
                                "temperature": 0.5,
                                "max_tokens": 1024
                            }
                        
                        headers = {
                            "Authorization": f"Bearer {GROQ_API_KEY}",
                            "Content-Type": "application/json"
                        }
                        
                        # Call Groq API
                        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
                        
                        if response.status_code == 200:
                            result = response.json()
                            ai_response = result['choices'][0]['message']['content']
                            
                            st.markdown("---")
                            st.subheader(t('analysis_results'))
                            st.markdown(ai_response)
                            
                            # Structured summary for medicines
                            if any(word in ai_response.lower() for word in ["medication", "medicine", "tablet", "syrup", "dosage"]):
                                st.success(f"Successfully extracted {t('extracted_medicines')}")
                        else:
                            st.error(f"API Error ({response.status_code}): {response.text}")
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

# Tab 12: Admin Dashboard (Restricted)
if show_admin and tab12:
    with tab12:
        st.header(t('admin_title'))
        st.subheader(t('registered_users'))
        
        users = auth_utils.get_all_users()
        if users:
            user_df = pd.DataFrame(users)
            # Reorder columns for display
            if 'username' in user_df.columns and 'email' in user_df.columns:
                user_df = user_df[['username', 'email']]
            
            st.table(user_df)
            st.info(f"Total Users: {len(users)}")
        else:
            st.warning("No users found or error connecting to database.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Medical Insurance Cost Predictor</strong> | Built with Streamlit & Machine Learning</p>
    <p>This tool provides estimates based on statistical models. Actual insurance costs may vary.</p>
</div>
""", unsafe_allow_html=True)
