import pandas as pd
import numpy as np
import streamlit as st
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Dummy classes to prevent NameError and basic crashes
    class RandomForestRegressor:
        def __init__(self, **kwargs): pass
        def fit(self, X, y): return self
        def score(self, X, y): return 0.0
        def predict(self, X): return np.zeros(len(X))
    class LabelEncoder:
        def __init__(self): pass
        def fit_transform(self, x): return x
        def transform(self, x): return x
    def train_test_split(*args, **kwargs):
        return args[0], args[0], args[1], args[1]
import pickle
import os
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def generate_medical_dataset(n_samples=1338):
    """
    Generate a synthetic medical cost dataset similar to the Kaggle dataset
    """
    np.random.seed(42)
    
    # Generate features
    ages = np.random.randint(18, 65, n_samples)
    sexes = np.random.choice(['male', 'female'], n_samples)
    bmis = np.random.normal(30, 6, n_samples)
    bmis = np.clip(bmis, 15, 50)  # Realistic BMI range
    children = np.random.choice([0, 1, 2, 3, 4, 5], n_samples, p=[0.4, 0.25, 0.2, 0.1, 0.04, 0.01])
    smokers = np.random.choice(['yes', 'no'], n_samples, p=[0.2, 0.8])
    regions = np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples)
    
    # Calculate charges with realistic relationships
    base_cost = 3000
    charges = []
    
    for i in range(n_samples):
        cost = base_cost
        
        # Age factor (increases with age)
        cost += ages[i] * 250
        
        # BMI factor
        if bmis[i] > 30:
            cost += (bmis[i] - 30) * 300
        else:
            cost += bmis[i] * 50
            
        # Smoker factor (major impact)
        if smokers[i] == 'yes':
            cost *= 2.5
            
        # Children factor
        cost += children[i] * 500
        
        # Regional variation
        regional_multipliers = {
            'northeast': 1.1,
            'northwest': 0.95,
            'southeast': 1.15,
            'southwest': 0.9
        }
        cost *= regional_multipliers[regions[i]]
        
        # Add some random variation
        cost *= np.random.uniform(0.85, 1.15)
        
        charges.append(round(cost, 2))
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': ages,
        'sex': sexes,
        'bmi': bmis.round(2),
        'children': children,
        'smoker': smokers,
        'region': regions,
        'charges': charges
    })
    
    return df

def train_model(df):
    """
    Train Random Forest and XGBoost models on the medical cost dataset
    Uses the best performing model based on test score
    """
    # Prepare features
    X = df.drop('charges', axis=1).copy()
    y = df['charges'].copy()
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()
    
    X['sex'] = le_sex.fit_transform(X['sex'])
    X['smoker'] = le_smoker.fit_transform(X['smoker'])
    X['region'] = le_region.fit_transform(X['region'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    rf_train_score = rf_model.score(X_train, y_train)
    rf_test_score = rf_model.score(X_test, y_test)
    
    # Initialize best model as Random Forest
    best_model = rf_model
    best_train_score = rf_train_score
    best_test_score = rf_test_score
    model_type = 'Random Forest'
    
    # Train XGBoost model if available
    xgb_test_score = None
    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            objective='reg:squarederror'
        )
        xgb_model.fit(X_train, y_train)
        
        xgb_train_score = xgb_model.score(X_train, y_train)
        xgb_test_score = xgb_model.score(X_test, y_test)
        
        # Use XGBoost if it performs better
        if xgb_test_score > rf_test_score:
            best_model = xgb_model
            best_train_score = xgb_train_score
            best_test_score = xgb_test_score
            model_type = 'XGBoost'
    
    # Save model and encoders
    model_data = {
        'model': best_model,
        'model_type': model_type,
        'le_sex': le_sex,
        'le_smoker': le_smoker,
        'le_region': le_region,
        'train_score': best_train_score,
        'test_score': best_test_score,
        'feature_names': list(X.columns),
        'rf_score': rf_test_score,
        'xgb_score': xgb_test_score
    }
    
    with open('insurance_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    return model_data

def load_model():
    """
    Load the trained model and encoders
    """
    if not os.path.exists('insurance_model.pkl'):
        # If model doesn't exist, load existing dataset and train
        if os.path.exists('insurance_data.csv'):
            df = pd.read_csv('insurance_data.csv')
            return train_model(df)
        else:
            raise FileNotFoundError("No model or dataset found. Please provide insurance_data.csv to train the model.")
    
    with open('insurance_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def predict_cost(model_data, age, sex, bmi, children, smoker, region):
    """
    Predict insurance cost for given parameters
    """
    # Encode inputs
    sex_encoded = model_data['le_sex'].transform([sex])[0]
    smoker_encoded = model_data['le_smoker'].transform([smoker])[0]
    region_encoded = model_data['le_region'].transform([region])[0]
    
    # Create feature array
    features = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    
    # Predict
    prediction = model_data['model'].predict(features)[0]
    
    return round(prediction, 2)

@st.cache_data
def get_risk_level(cost):
    """
    Categorize risk level based on predicted cost
    """
    if cost < 5000:
        return "Low", "🟢"
    elif cost < 15000:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

@st.cache_data
def get_govt_vs_private_comparison(predicted_cost):
    """
    Compare with government and private insurance ranges
    """
    # Government schemes typically cover basic costs
    govt_coverage = min(predicted_cost * 0.6, 5000)
    
    # Private insurance offers more comprehensive coverage
    private_base = predicted_cost * 0.85
    private_premium = predicted_cost * 1.1
    
    return {
        'govt_coverage': round(govt_coverage, 2),
        'govt_out_of_pocket': round(predicted_cost - govt_coverage, 2),
        'private_base': round(private_base, 2),
        'private_premium': round(private_premium, 2)
    }

def generate_pdf_report(user_data, predicted_cost, risk_level, comparison_data, factor_impacts=None):
    """
    Generate a PDF report with prediction summary and analysis
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from datetime import datetime
    import io
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("Medical Insurance Cost Prediction Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Date
    date_str = datetime.now().strftime("%B %d, %Y")
    story.append(Paragraph(f"Generated on: {date_str}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Personal Information
    story.append(Paragraph("Personal Information", heading_style))
    personal_data = [
        ['Age:', f"{user_data['age']} years"],
        ['Gender:', user_data['sex'].capitalize()],
        ['BMI:', f"{user_data['bmi']:.1f}"],
        ['Children:', str(user_data['children'])],
        ['Smoking Status:', user_data['smoker'].capitalize()],
        ['Region:', user_data['region'].capitalize()]
    ]
    
    personal_table = Table(personal_data, colWidths=[2*inch, 4*inch])
    personal_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(personal_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Prediction Results
    story.append(Paragraph("Prediction Results", heading_style))
    
    risk_color = '#27ae60' if risk_level == 'Low' else ('#f39c12' if risk_level == 'Medium' else '#e74c3c')
    
    prediction_data = [
        ['Predicted Annual Cost:', f"₹{predicted_cost:,.2f}"],
        ['Monthly Premium (Est.):', f"₹{predicted_cost/12:,.2f}"],
        ['Risk Level:', risk_level]
    ]
    
    prediction_table = Table(prediction_data, colWidths=[2.5*inch, 3.5*inch])
    prediction_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('TEXTCOLOR', (1, 0), (1, 0), colors.HexColor('#2980b9')),
        ('FONTSIZE', (1, 0), (1, 0), 14),
        ('TEXTCOLOR', (1, 2), (1, 2), colors.HexColor(risk_color)),
        ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(prediction_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Cost Comparison
    story.append(Paragraph("Government vs Private Insurance Comparison", heading_style))
    
    comparison_info = [
        ['Insurance Type', 'Coverage', 'Cost'],
        ['Government Scheme', f"₹{comparison_data['govt_coverage']:,.2f}", f"Out-of-pocket: ₹{comparison_data['govt_out_of_pocket']:,.2f}"],
        ['Private (Base Plan)', f"₹{comparison_data['private_base']:,.2f}", f"Est. ₹{comparison_data['private_base']:,.2f}"],
        ['Private (Premium Plan)', f"₹{comparison_data['private_premium']:,.2f}", f"Est. ₹{comparison_data['private_premium']:,.2f}"]
    ]
    
    comparison_table = Table(comparison_info, colWidths=[2*inch, 2*inch, 2*inch])
    comparison_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ]))
    story.append(comparison_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Cost Factor Analysis
    if factor_impacts:
        story.append(Paragraph("Cost Factor Analysis", heading_style))
        
        factor_data = [['Factor', 'Impact (₹)']]
        for factor, impact in factor_impacts.items():
            factor_data.append([factor, f"₹{impact:,.2f}"])
        
        factor_table = Table(factor_data, colWidths=[3*inch, 2*inch])
        factor_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ]))
        story.append(factor_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Recommendations
    story.append(Paragraph("Recommendations", heading_style))
    recommendations = []
    
    if user_data['smoker'] == 'yes':
        recommendations.append("• Consider quitting smoking to reduce insurance costs by up to 150-250%")
    
    if user_data['bmi'] > 30:
        recommendations.append("• A weight management program could help reduce your BMI and lower premiums")
    
    if risk_level == 'High':
        recommendations.append("• Explore comprehensive insurance options to ensure adequate coverage")
        recommendations.append("• Consider both government subsidies and private insurance plans")
    
    if comparison_data['govt_out_of_pocket'] < comparison_data['private_base']:
        recommendations.append("• Government healthcare scheme may be more economical for your profile")
    else:
        recommendations.append("• Private insurance might offer better value with comprehensive coverage")
    
    if not recommendations:
        recommendations.append("• Your insurance profile is healthy. Maintain your current lifestyle")
        recommendations.append("• Review insurance options annually for the best rates")
    
    for rec in recommendations:
        story.append(Paragraph(rec, styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#7f8c8d'),
        alignment=TA_CENTER
    )
    story.append(Paragraph(
        "This report provides estimates based on statistical models. Actual insurance costs may vary. "
        "Please consult with insurance providers for accurate quotes.",
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

@st.cache_data
def estimate_accident_injury_cost(accident_type, severity, hospitalization, surgery, recovery_days):
    """
    Estimate additional insurance cost for accident/injury
    
    Parameters:
    - accident_type: car accident, fall, sports injury, workplace injury, other
    - severity: minor, moderate, severe, critical
    - hospitalization: yes/no
    - surgery: yes/no
    - recovery_days: number of days for recovery
    """
    base_cost = 0
    
    # Base cost by accident type
    accident_costs = {
        'car accident': 15000,
        'fall': 8000,
        'sports injury': 10000,
        'workplace injury': 12000,
        'other': 7000
    }
    base_cost += accident_costs.get(accident_type, 7000)
    
    # Severity multiplier
    severity_multipliers = {
        'minor': 0.5,
        'moderate': 1.0,
        'severe': 2.0,
        'critical': 3.5
    }
    base_cost *= severity_multipliers.get(severity, 1.0)
    
    # Hospitalization cost
    if hospitalization == 'yes':
        base_cost += 5000
        base_cost += (recovery_days * 1500)
    
    # Surgery cost
    if surgery == 'yes':
        base_cost += 25000
    
    # Recovery time factor (ongoing care, medication, therapy)
    recovery_cost = recovery_days * 100
    base_cost += recovery_cost
    
    return round(base_cost, 2)

@st.cache_data
def get_accident_cost_breakdown(accident_type, severity, hospitalization, surgery, recovery_days):
    """
    Get detailed breakdown of accident/injury costs
    """
    breakdown = {}
    
    # Base cost by accident type
    accident_costs = {
        'car accident': 15000,
        'fall': 8000,
        'sports injury': 10000,
        'workplace injury': 12000,
        'other': 7000
    }
    base = accident_costs.get(accident_type, 7000)
    
    # Severity multiplier
    severity_multipliers = {
        'minor': 0.5,
        'moderate': 1.0,
        'severe': 2.0,
        'critical': 3.5
    }
    severity_mult = severity_multipliers.get(severity, 1.0)
    
    breakdown['Base Treatment Cost'] = base * severity_mult
    
    if hospitalization == 'yes':
        breakdown['Hospitalization'] = 5000
        breakdown['Daily Hospital Care'] = recovery_days * 1500
    
    if surgery == 'yes':
        breakdown['Surgery'] = 25000
    
    breakdown['Recovery & Medication'] = recovery_days * 100
    
    return breakdown

@st.cache_data
def get_government_scheme_recommendations(age, children, smoker, predicted_cost, bmi, region):
    """
    Recommend government healthcare schemes based on user profile and predicted cost
    """
    recommendations = []
    
    # Universal Healthcare Program
    if predicted_cost < 10000:
        recommendations.append({
            'name': 'Basic Healthcare Assistance Program',
            'eligibility': 'All residents with annual healthcare costs below ₹10,000',
            'coverage': 'Up to ₹5,000 annual coverage for basic healthcare needs',
            'benefits': [
                'Primary care visits covered',
                'Preventive care and vaccinations',
                'Generic prescription medications',
                'Basic diagnostic tests'
            ],
            'application': 'Apply online at your state healthcare marketplace',
            'priority': 'High'
        })
    
    # Low-Income Healthcare Subsidy
    if predicted_cost > 15000:
        recommendations.append({
            'name': 'Healthcare Cost Relief Program',
            'eligibility': 'Individuals with high medical costs (>₹15,000 annually)',
            'coverage': 'Subsidized premiums and reduced out-of-pocket costs',
            'benefits': [
                'Premium subsidies up to 80%',
                'Reduced deductibles and copays',
                'Coverage for chronic condition management',
                'Prescription drug assistance'
            ],
            'application': 'Contact your local health department or apply online',
            'priority': 'High'
        })
    
    # Family Healthcare Support
    if children >= 2:
        recommendations.append({
            'name': 'Family Healthcare Support Program',
            'eligibility': 'Families with 2 or more dependent children',
            'coverage': f'Coverage for family of {children + 1} members',
            'benefits': [
                'Pediatric care coverage',
                'Maternity and newborn care',
                'Family dental and vision',
                'Mental health services for children'
            ],
            'application': 'Apply through state family services department',
            'priority': 'Medium'
        })
    
    # Senior Healthcare Program
    if age >= 55:
        recommendations.append({
            'name': 'Senior Health Assistance Program',
            'eligibility': 'Adults aged 55 and older',
            'coverage': 'Comprehensive coverage for age-related health needs',
            'benefits': [
                'Annual health screenings',
                'Chronic disease management',
                'Prescription drug coverage',
                'Home healthcare services',
                'Preventive care services'
            ],
            'application': 'Enroll through senior services office or online portal',
            'priority': 'High'
        })
    
    # Smoking Cessation Program
    if smoker == 'yes':
        recommendations.append({
            'name': 'Tobacco Cessation Support Program',
            'eligibility': 'Current smokers seeking to quit',
            'coverage': 'Free cessation support and medications',
            'benefits': [
                'Nicotine replacement therapy',
                'Counseling and support groups',
                'Prescription cessation medications',
                'Follow-up care for 12 months',
                'Can reduce insurance costs by 150-250%'
            ],
            'application': 'Call the quitline or visit cessation program website',
            'priority': 'High'
        })
    
    # Weight Management Program
    if bmi > 30:
        recommendations.append({
            'name': 'Healthy Weight Initiative',
            'eligibility': 'Individuals with BMI > 30',
            'coverage': 'Free weight management and nutrition services',
            'benefits': [
                'Nutritionist consultations',
                'Fitness program access',
                'Weight loss medications (if medically necessary)',
                'Diabetes prevention program',
                'Can reduce insurance premiums'
            ],
            'application': 'Enroll through primary care physician or health department',
            'priority': 'Medium'
        })
    
    # Regional Health Programs
    if region in ['southeast', 'southwest']:
        recommendations.append({
            'name': 'Regional Health Access Program',
            'eligibility': f'Residents of {region} region',
            'coverage': 'Enhanced access to regional healthcare facilities',
            'benefits': [
                'Network of community health centers',
                'Telehealth services',
                'Mobile health clinics',
                'Sliding scale fees based on income'
            ],
            'application': 'Contact regional health authority',
            'priority': 'Medium'
        })
    
    # Preventive Care Program (for everyone)
    recommendations.append({
        'name': 'National Preventive Care Initiative',
        'eligibility': 'All residents',
        'coverage': 'Free preventive care services',
        'benefits': [
            'Annual wellness exam',
            'Cancer screenings',
            'Immunizations',
            'Blood pressure and cholesterol checks',
            'Mental health screening'
        ],
        'application': 'Available at all participating healthcare providers',
        'priority': 'Medium'
    })
    
    # Sort by priority
    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    recommendations.sort(key=lambda x: priority_order.get(x['priority'], 2))
    
    return recommendations
