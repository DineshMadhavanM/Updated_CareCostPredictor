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
translations = {
    'en': {
        'page_title': 'Medical Insurance Cost Predictor',
        'main_title': '🏥 Medical Insurance Cost Predictor',
        'main_description': '''This application uses advanced Machine Learning (Random Forest and XGBoost) to predict medical insurance costs 
based on personal health and demographic factors. Explore how different factors affect insurance costs 
and compare government vs private insurance options.''',
        'language_selector': 'Language / भाषा',
        'model_info': '📊 Model Information',
        'model_type': 'Model Type',
        'training_accuracy': 'Training Accuracy',
        'testing_accuracy': 'Testing Accuracy',
        'dataset_size': 'Dataset Size',
        'samples': 'samples',
        'model_comparison': '🏆 Model Comparison',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'xgb_improved': 'XGBoost improved by',
        'rf_better': 'Random Forest performed better by',
        'models_equal': 'Both models performed equally',
        'dataset_stats': '📈 Dataset Statistics',
        'age_range': 'Age Range',
        'years': 'years',
        'bmi_range': 'BMI Range',
        'avg_cost': 'Avg Cost',
        'smokers': 'Smokers',
        'export_data': '📥 Export Data',
        'predictions_made': 'Predictions Made',
        'download_csv': '📊 Download CSV',
        'clear_history': '🗑️ Clear History',
        'no_predictions': 'No predictions yet',
        'tab_prediction': '🔮 Prediction',
        'tab_visualizations': '📊 Visualizations',
        'tab_whatif': '🔄 What-If Analysis',
        'tab_cost_comparison': '💰 Cost Comparison',
        'tab_accident': '🚑 Accident/Injury Cost',
        'insurance_cost_prediction': 'Insurance Cost Prediction',
        'personal_info': 'Personal Information',
        'age': 'Age',
        'age_help': 'Your current age in years',
        'gender': 'Gender',
        'male': 'male',
        'female': 'female',
        'children': 'Number of Children',
        'region': 'Region',
        'northeast': 'northeast',
        'northwest': 'northwest',
        'southeast': 'southeast',
        'southwest': 'southwest',
        'health_info': 'Health Information',
        'bmi': 'BMI (Body Mass Index)',
        'bmi_help': 'BMI = weight(kg) / height(m)²',
        'smoking_status': 'Smoking Status',
        'yes': 'yes',
        'no': 'no',
        'bmi_category': 'BMI Category',
        'underweight': 'Underweight',
        'normal_weight': 'Normal weight',
        'overweight': 'Overweight',
        'obese': 'Obese',
        'predict_button': '🔮 Predict Insurance Cost',
        'prediction_results': 'Prediction Results',
        'predicted_annual_cost': 'Predicted Annual Cost',
        'risk_level': 'Risk Level',
        'monthly_premium': 'Monthly Premium (Est.)',
        'cost_factor_analysis': 'Cost Factor Analysis',
        'age_factor': 'Age Factor',
        'bmi_factor': 'BMI Factor',
        'smoking_factor': 'Smoking Factor',
        'children_factor': 'Children Factor',
        'factor_impact_title': 'Estimated Impact of Each Factor on Cost',
        'export_report': 'Export Report',
        'download_pdf': '📄 Download PDF Report',
        'interactive_visualizations': 'Interactive Data Visualizations',
        'cost_vs_age': 'Insurance Cost vs Age',
        'insurance_cost': 'Insurance Cost (₹)',
        'age_years': 'Age (years)',
        'login': 'Login',
        'signup': 'Sign Up',
        'username': 'Username',
        'password': 'Password',
        'logout': 'Logout',
        'auth_welcome': 'Welcome to CareCost Predictor',
        'auth_error': 'Invalid username or password',
        'auth_success': 'Successfully logged in!',
        'no_account': "Don't have an account?",
        'have_account': 'Already have an account?',
        'create_account': 'Create Account',
        'email': 'Email ID',
        'confirm_password': 'Confirm Password',
        'passwords_dont_match': 'Passwords do not match',
        'tab_admin': '🔑 Admin Dashboard',
        'admin_title': 'Administrative Dashboard',
        'registered_users': '👥 Registered Users',
        'username': 'Username',
        'email_label': 'Email ID',
        'avg_cost_children': 'Average Insurance Cost by Number of Children',
        'average_cost': 'Average Cost (₹)',
        'number_of_children': 'Number of Children',
        'cost_vs_bmi': 'Insurance Cost vs BMI',
        'smoking_impact': 'Smoking Status Impact on Insurance Cost',
        'smoker': 'Smoker',
        'regional_cost_analysis': 'Regional Cost Analysis',
        'average': 'Average',
        'minimum': 'Minimum',
        'maximum': 'Maximum',
        'whatif_tool': 'What-If Analysis Tool',
        'whatif_description': 'Adjust parameters below to see how changes affect your insurance cost prediction',
        'baseline_scenario': '📍 Baseline Scenario',
        'baseline_age': 'Baseline Age',
        'baseline_gender': 'Baseline Gender',
        'baseline_bmi': 'Baseline BMI',
        'baseline_children': 'Baseline Children',
        'baseline_smoker': 'Baseline Smoker',
        'baseline_region': 'Baseline Region',
        'baseline_cost': 'Baseline Cost',
        'whatif_scenario': '🔄 What-If Scenario',
        'whatif_age': 'What-If Age',
        'whatif_gender': 'What-If Gender',
        'whatif_bmi': 'What-If BMI',
        'whatif_children': 'What-If Children',
        'whatif_smoker': 'What-If Smoker',
        'whatif_region': 'What-If Region',
        'whatif_cost': 'What-If Cost',
        'scenario_comparison': 'Scenario Comparison',
        'comparison_title': 'Cost Comparison: Baseline vs What-If',
        'baseline': 'Baseline',
        'whatif': 'What-If',
        'parameter_changes': 'Parameter Changes',
        'no_changes': 'No parameters changed. Adjust values to see the impact.',
        'govt_vs_private': 'Government vs Private Insurance Comparison',
        'govt_vs_private_desc': '''Compare estimated costs between government healthcare schemes and private insurance options.
Government schemes typically provide basic coverage with lower premiums, while private insurance 
offers comprehensive coverage with higher premiums.''',
        'enter_details': 'Enter Your Details',
        'compare_button': '💰 Compare Insurance Options',
        'comparison_results': 'Cost Comparison Results',
        'govt_scheme': '🏛️ Government Scheme',
        'govt_coverage': 'Government Coverage',
        'out_of_pocket': 'Your Out-of-Pocket',
        'coverage_percentage': 'Coverage Percentage',
        'pros': 'Pros:',
        'cons': 'Cons:',
        'govt_pro1': 'Lower premiums',
        'govt_pro2': 'Basic coverage included',
        'govt_pro3': 'Government subsidized',
        'govt_con1': 'Limited coverage',
        'govt_con2': 'Higher out-of-pocket costs',
        'govt_con3': 'Fewer hospital choices',
        'private_insurance': '🏥 Private Insurance',
        'base_plan_cost': 'Base Plan Cost',
        'premium_plan_cost': 'Premium Plan Cost',
        'private_pro1': 'Comprehensive coverage',
        'private_pro2': 'Wide hospital network',
        'private_pro3': 'Additional benefits',
        'private_con1': 'Higher premiums',
        'private_con2': 'Complex terms',
        'private_con3': 'Waiting periods',
        'visual_breakdown': 'Visual Cost Breakdown',
        'insurance_comparison': 'Insurance Cost Comparison',
        'govt_coverage_label': 'Government\nCoverage',
        'govt_oop_label': 'Government\nOut-of-Pocket',
        'private_base_label': 'Private\nBase Plan',
        'private_premium_label': 'Private\nPremium Plan',
        'government': 'Government',
        'private': 'Private',
        'predicted_total': 'Predicted Total Cost',
        'recommendations': '💡 Recommendations',
        'govt_economical': '✅ Government scheme may be more economical if you can manage the out-of-pocket costs.',
        'private_better': 'ℹ️ Private insurance might offer better value with comprehensive coverage.',
        'smoker_warning': '⚠️ As a smoker, consider quitting to significantly reduce insurance costs. Smoking can increase costs by 150-250%.',
        'bmi_warning': '⚠️ High BMI increases insurance costs. Consider a weight management program to reduce premiums.',
        'eligible_schemes': '🏛️ Eligible Government Healthcare Schemes',
        'schemes_description': 'Based on your profile, you may be eligible for the following government assistance programs:',
        'high_priority': 'High',
        'medium_priority': 'Medium',
        'priority': 'Priority',
        'eligibility': 'Eligibility',
        'coverage': 'Coverage',
        'benefits': 'Benefits',
        'how_to_apply': 'How to Apply',
        'highly_recommended': '✅ This program is highly recommended for your profile',
        'qualify_info': 'You qualify for {count} government healthcare programs. Consider applying to maximize your coverage and reduce out-of-pocket costs.',
        'accident_estimation': 'Accident/Injury Cost Estimation',
        'accident_description': '''Estimate additional insurance costs for accidents or injuries. This helps you understand potential 
out-of-pocket expenses and plan for unexpected medical events.''',
        'your_profile': 'Your Profile',
        'accident_details': 'Accident/Injury Details',
        'accident_type': 'Type of Accident/Injury',
        'accident_type_help': 'Select the type of accident or injury',
        'car_accident': 'car accident',
        'fall': 'fall',
        'sports_injury': 'sports injury',
        'workplace_injury': 'workplace injury',
        'other': 'other',
        'severity': 'Severity Level',
        'severity_help': 'Minor: cuts, bruises | Moderate: sprains, minor fractures | Severe: major fractures, internal injuries | Critical: life-threatening',
        'minor': 'minor',
        'moderate': 'moderate',
        'severe': 'severe',
        'critical': 'critical',
        'recovery_time': 'Estimated Recovery Time (days)',
        'recovery_help': 'Number of days needed for full recovery',
        'hospitalization': 'Hospitalization Required?',
        'hospitalization_help': 'Will you need to stay in the hospital?',
        'surgery': 'Surgery Required?',
        'surgery_help': 'Will surgical intervention be necessary?',
        'recovery_period': 'Recovery Period',
        'days': 'days',
        'months': 'months',
        'estimate_accident_button': '💉 Estimate Accident/Injury Cost',
        'cost_estimation_results': 'Cost Estimation Results',
        'base_annual_insurance': 'Base Annual Insurance',
        'accident_injury_cost': 'Accident/Injury Cost',
        'total_cost': 'Total Cost',
        'cost_increase': 'Cost Increase',
        'cost_breakdown': 'Cost Breakdown',
        'detailed_breakdown': 'Detailed Cost Breakdown',
        'component': 'Component',
        'financial_planning': '💰 Financial Planning',
        'immediate_costs': 'Immediate Costs',
        'emergency_treatment': 'Emergency Treatment',
        'hospital_stay': 'Hospital Stay',
        'surgery_cost': 'Surgery',
        'ongoing_costs': 'Ongoing Costs',
        'daily_care': 'Daily Care',
        'recovery_medication': 'Recovery & Medication',
        'monthly_average': 'Monthly Average',
        'tab_trends': '📈 Cost Trends',
        'tab_chatbot': '🤖 AI Chatbot',
        'tab_document': '📄 Document Analyzer',
        'tab_quotes': '💵 Real-time Quotes',
        'tab_tax': '🧾 Tax Benefits',
        'cost_trends_dashboard': 'Cost Trends Dashboard',
        'trends_description': 'Track how your predicted insurance costs change over time based on your prediction history',
        'trend_over_time': 'Cost Trend Over Time',
        'predictions_count': 'Total Predictions',
        'avg_predicted_cost': 'Average Predicted Cost',
        'cost_range': 'Cost Range',
        'highest_cost': 'Highest Cost',
        'lowest_cost': 'Lowest Cost',
        'cost_by_age_group': 'Average Cost by Age Group',
        'cost_by_smoker': 'Cost Distribution by Smoking Status',
        'no_trends_data': 'No prediction history available. Make some predictions to see trends!',
        'ai_chatbot': 'AI Insurance Advisor',
        'chatbot_description': 'Ask me anything about health insurance, coverage, premiums, or get personalized advice',
        'ask_question': 'Ask your question',
        'send': 'Send',
        'chatbot_placeholder': 'E.g., What is the difference between health and life insurance?',
        'chatbot_thinking': 'Thinking...',
        'setup_ai': 'Setup AI Integration',
        'ai_not_configured': 'AI chatbot is not configured. Please set up OpenAI integration to use this feature.',
        'document_analyzer': 'Insurance Policy Document Analyzer',
        'doc_description': 'Upload your insurance policy PDF to get AI-powered analysis and insights',
        'upload_policy': 'Upload Insurance Policy (PDF)',
        'analyze_button': '🔍 Analyze Document',
        'analyzing': 'Analyzing document...',
        'analysis_results': 'Analysis Results',
        'key_points': 'Key Points',
        'coverage_details': 'Coverage Details',
        'exclusions': 'Exclusions',
        'premium_info': 'Premium Information',
        'no_document': 'Please upload a PDF document to analyze',
        'realtime_quotes': 'Real-time Insurance Quotes',
        'quotes_description': 'Get instant insurance quotes from multiple providers based on your profile',
        'get_quotes': '💰 Get Insurance Quotes',
        'fetching_quotes': 'Fetching quotes from providers...',
        'available_plans': 'Available Insurance Plans',
        'provider': 'Provider',
        'plan_name': 'Plan Name',
        'annual_premium': 'Annual Premium',
        'coverage_amount': 'Coverage Amount',
        'key_features': 'Key Features',
        'compare_plans': 'Compare Plans',
        'quotes_disclaimer': 'Note: These are estimated quotes. Actual premiums may vary based on medical underwriting.',
        'tax_calculator': 'Insurance Tax Benefit Calculator',
        'tax_description': 'Calculate tax deductions under Section 80D of the Income Tax Act',
        'premium_paid': 'Annual Premium Paid (₹)',
        'age_category': 'Age Category',
        'below_60': 'Below 60 years',
        'above_60': 'Above 60 years (Senior Citizen)',
        'parents_premium': 'Parents Premium Paid (₹)',
        'parents_age': 'Parents Age Category',
        'preventive_checkup': 'Preventive Health Checkup Cost (₹)',
        'calculate_tax': '🧾 Calculate Tax Benefit',
        'tax_benefit_results': 'Tax Benefit Summary',
        'self_deduction': 'Self/Family Deduction',
        'parents_deduction': 'Parents Deduction',
        'checkup_deduction': 'Health Checkup Deduction',
        'total_deduction': 'Total Deduction (80D)',
        'tax_saved_30': 'Tax Saved (30% bracket)',
        'tax_saved_20': 'Tax Saved (20% bracket)',
        'tax_saved_10': 'Tax Saved (10% bracket)',
        'section_80d_info': '💡 Section 80D Information',
        'deduction_limits': 'Deduction Limits',
        'self_limit': 'Self/Spouse/Children: ₹25,000 (₹50,000 if senior citizen)',
        'parents_limit': 'Parents: ₹25,000 (₹50,000 if senior citizen)',
        'checkup_limit': 'Preventive Health Checkup: ₹5,000 (included in above limits)',
        'max_deduction': 'Maximum Total Deduction: ₹1,00,000',
        'tab_receipt_analyzer': '🧾 Receipt Analyzer',
        'receipt_analyzer_title': 'Medical Receipt & Description Analyzer',
        'receipt_analyzer_desc': "Upload your medical receipts or doctor's prescriptions (Image or PDF) to extract medicines, instructions, and key medical details.",
        'upload_receipt': 'Upload Receipt or Prescription',
        'analyze_receipt_button': '🔍 Analyze Receipt',
        'extracted_medicines': '💊 Extracted Medicines',
        'doctor_instructions': "👨‍⚕️ Doctor's Instructions",
        'important_details': 'ℹ️ Important Details'
    },
    'hi': {
        'page_title': 'चिकित्सा बीमा लागत अनुमानक',
        'main_title': '🏥 चिकित्सा बीमा लागत अनुमानक',
        'main_description': '''यह एप्लिकेशन व्यक्तिगत स्वास्थ्य और जनसांख्यिकीय कारकों के आधार पर चिकित्सा बीमा लागत की भविष्यवाणी करने के लिए उन्नत मशीन लर्निंग (रैंडम फॉरेस्ट और XGBoost) का उपयोग करता है। जानें कि विभिन्न कारक बीमा लागत को कैसे प्रभावित करते हैं 
और सरकारी बनाम निजी बीमा विकल्पों की तुलना करें।''',
        'language_selector': 'भाषा / Language',
        'model_info': '📊 मॉडल जानकारी',
        'model_type': 'मॉडल प्रकार',
        'training_accuracy': 'प्रशिक्षण सटीकता',
        'testing_accuracy': 'परीक्षण सटीकता',
        'dataset_size': 'डेटासेट आकार',
        'samples': 'नमूने',
        'model_comparison': '🏆 मॉडल तुलना',
        'random_forest': 'रैंडम फॉरेस्ट',
        'xgboost': 'XGBoost',
        'xgb_improved': 'XGBoost में सुधार',
        'rf_better': 'रैंडम फॉरेस्ट बेहतर प्रदर्शन',
        'models_equal': 'दोनों मॉडल समान रूप से प्रदर्शन किया',
        'dataset_stats': '📈 डेटासेट आँकड़े',
        'age_range': 'आयु सीमा',
        'years': 'वर्ष',
        'bmi_range': 'BMI सीमा',
        'avg_cost': 'औसत लागत',
        'smokers': 'धूम्रपान करने वाले',
        'export_data': '📥 डेटा निर्यात करें',
        'predictions_made': 'की गई भविष्यवाणियाँ',
        'download_csv': '📊 CSV डाउनलोड करें',
        'clear_history': '🗑️ इतिहास साफ़ करें',
        'no_predictions': 'अभी तक कोई भविष्यवाणी नहीं',
        'tab_prediction': '🔮 भविष्यवाणी',
        'tab_visualizations': '📊 विज़ुअलाइज़ेशन',
        'tab_whatif': '🔄 यदि-तो विश्लेषण',
        'tab_cost_comparison': '💰 लागत तुलना',
        'tab_accident': '🚑 दुर्घटना/चोट लागत',
        'insurance_cost_prediction': 'बीमा लागत भविष्यवाणी',
        'personal_info': 'व्यक्तिगत जानकारी',
        'age': 'आयु',
        'age_help': 'वर्षों में आपकी वर्तमान आयु',
        'gender': 'लिंग',
        'male': 'पुरुष',
        'female': 'महिला',
        'children': 'बच्चों की संख्या',
        'region': 'क्षेत्र',
        'northeast': 'पूर्वोत्तर',
        'northwest': 'उत्तर पश्चिम',
        'southeast': 'दक्षिण पूर्व',
        'southwest': 'दक्षिण पश्चिम',
        'health_info': 'स्वास्थ्य जानकारी',
        'bmi': 'BMI (बॉडी मास इंडेक्स)',
        'bmi_help': 'BMI = वजन(किग्रा) / ऊंचाई(मी)²',
        'smoking_status': 'धूम्रपान की स्थिति',
        'yes': 'हाँ',
        'no': 'नहीं',
        'bmi_category': 'BMI श्रेणी',
        'underweight': 'कम वजन',
        'normal_weight': 'सामान्य वजन',
        'overweight': 'अधिक वजन',
        'obese': 'मोटापा',
        'predict_button': '🔮 बीमा लागत की भविष्यवाणी करें',
        'prediction_results': 'भविष्यवाणी परिणाम',
        'predicted_annual_cost': 'अनुमानित वार्षिक लागत',
        'risk_level': 'जोखिम स्तर',
        'monthly_premium': 'मासिक प्रीमियम (अनुमानित)',
        'cost_factor_analysis': 'लागत कारक विश्लेषण',
        'age_factor': 'आयु कारक',
        'bmi_factor': 'BMI कारक',
        'smoking_factor': 'धूम्रपान कारक',
        'children_factor': 'बच्चे कारक',
        'factor_impact_title': 'प्रत्येक कारक का लागत पर अनुमानित प्रभाव',
        'export_report': 'रिपोर्ट निर्यात करें',
        'download_pdf': '📄 PDF रिपोर्ट डाउनलोड करें',
        'interactive_visualizations': 'इंटरैक्टिव डेटा विज़ुअलाइज़ेशन',
        'cost_vs_age': 'बीमा लागत बनाम आयु',
        'insurance_cost': 'बीमा लागत (₹)',
        'age_years': 'आयु (वर्ष)',
        'avg_cost_children': 'बच्चों की संख्या के अनुसार औसत बीमा लागत',
        'average_cost': 'औसत लागत (₹)',
        'number_of_children': 'बच्चों की संख्या',
        'cost_vs_bmi': 'बीमा लागत बनाम BMI',
        'smoking_impact': 'बीमा लागत पर धूम्रपान स्थिति का प्रभाव',
        'smoker': 'धूम्रपान करने वाला',
        'regional_cost_analysis': 'क्षेत्रीय लागत विश्लेषण',
        'average': 'औसत',
        'minimum': 'न्यूनतम',
        'maximum': 'अधिकतम',
        'whatif_tool': 'यदि-तो विश्लेषण उपकरण',
        'whatif_description': 'यह देखने के लिए नीचे पैरामीटर समायोजित करें कि परिवर्तन आपकी बीमा लागत भविष्यवाणी को कैसे प्रभावित करते हैं',
        'baseline_scenario': '📍 आधार परिदृश्य',
        'baseline_age': 'आधार आयु',
        'baseline_gender': 'आधार लिंग',
        'baseline_bmi': 'आधार BMI',
        'baseline_children': 'आधार बच्चे',
        'baseline_smoker': 'आधार धूम्रपान',
        'baseline_region': 'आधार क्षेत्र',
        'baseline_cost': 'आधार लागत',
        'whatif_scenario': '🔄 यदि-तो परिदृश्य',
        'whatif_age': 'यदि-तो आयु',
        'whatif_gender': 'यदि-तो लिंग',
        'whatif_bmi': 'यदि-तो BMI',
        'whatif_children': 'यदि-तो बच्चे',
        'whatif_smoker': 'यदि-तो धूम्रपान',
        'whatif_region': 'यदि-तो क्षेत्र',
        'whatif_cost': 'यदि-तो लागत',
        'scenario_comparison': 'परिदृश्य तुलना',
        'comparison_title': 'लागत तुलना: आधार बनाम यदि-तो',
        'baseline': 'आधार',
        'whatif': 'यदि-तो',
        'parameter_changes': 'पैरामीटर परिवर्तन',
        'no_changes': 'कोई पैरामीटर नहीं बदला। प्रभाव देखने के लिए मान समायोजित करें।',
        'govt_vs_private': 'सरकारी बनाम निजी बीमा तुलना',
        'govt_vs_private_desc': '''सरकारी स्वास्थ्य योजनाओं और निजी बीमा विकल्पों के बीच अनुमानित लागत की तुलना करें।
सरकारी योजनाएं आमतौर पर कम प्रीमियम के साथ बुनियादी कवरेज प्रदान करती हैं, जबकि निजी बीमा 
उच्च प्रीमियम के साथ व्यापक कवरेज प्रदान करता है।''',
        'enter_details': 'अपना विवरण दर्ज करें',
        'compare_button': '💰 बीमा विकल्पों की तुलना करें',
        'comparison_results': 'लागत तुलना परिणाम',
        'govt_scheme': '🏛️ सरकारी योजना',
        'govt_coverage': 'सरकारी कवरेज',
        'out_of_pocket': 'आपकी जेब से',
        'coverage_percentage': 'कवरेज प्रतिशत',
        'pros': 'फायदे:',
        'cons': 'नुकसान:',
        'govt_pro1': 'कम प्रीमियम',
        'govt_pro2': 'बुनियादी कवरेज शामिल',
        'govt_pro3': 'सरकारी सब्सिडी',
        'govt_con1': 'सीमित कवरेज',
        'govt_con2': 'अधिक जेब से खर्च',
        'govt_con3': 'कम अस्पताल विकल्प',
        'private_insurance': '🏥 निजी बीमा',
        'base_plan_cost': 'बेस प्लान लागत',
        'premium_plan_cost': 'प्रीमियम प्लान लागत',
        'private_pro1': 'व्यापक कवरेज',
        'private_pro2': 'व्यापक अस्पताल नेटवर्क',
        'private_pro3': 'अतिरिक्त लाभ',
        'private_con1': 'उच्च प्रीमियम',
        'private_con2': 'जटिल शर्तें',
        'private_con3': 'प्रतीक्षा अवधि',
        'visual_breakdown': 'दृश्य लागत विवरण',
        'insurance_comparison': 'बीमा लागत तुलना',
        'govt_coverage_label': 'सरकारी\nकवरेज',
        'govt_oop_label': 'सरकारी\nजेब से',
        'private_base_label': 'निजी\nबेस प्लान',
        'private_premium_label': 'निजी\nप्रीमियम प्लान',
        'government': 'सरकारी',
        'private': 'निजी',
        'predicted_total': 'अनुमानित कुल लागत',
        'recommendations': '💡 सिफारिशें',
        'govt_economical': '✅ यदि आप जेब से खर्च प्रबंधित कर सकते हैं तो सरकारी योजना अधिक किफायती हो सकती है।',
        'private_better': 'ℹ️ निजी बीमा व्यापक कवरेज के साथ बेहतर मूल्य प्रदान कर सकता है।',
        'smoker_warning': '⚠️ धूम्रपान करने वाले के रूप में, बीमा लागत को काफी कम करने के लिए छोड़ने पर विचार करें। धूम्रपान से लागत 150-250% बढ़ सकती है।',
        'bmi_warning': '⚠️ उच्च BMI बीमा लागत बढ़ाता है। प्रीमियम कम करने के लिए वजन प्रबंधन कार्यक्रम पर विचार करें।',
        'eligible_schemes': '🏛️ पात्र सरकारी स्वास्थ्य योजनाएं',
        'schemes_description': 'आपकी प्रोफ़ाइल के आधार पर, आप निम्नलिखित सरकारी सहायता कार्यक्रमों के लिए पात्र हो सकते हैं:',
        'high_priority': 'उच्च',
        'medium_priority': 'मध्यम',
        'priority': 'प्राथमिकता',
        'eligibility': 'पात्रता',
        'coverage': 'कवरेज',
        'benefits': 'लाभ',
        'how_to_apply': 'आवेदन कैसे करें',
        'highly_recommended': '✅ यह कार्यक्रम आपकी प्रोफ़ाइल के लिए अत्यधिक अनुशंसित है',
        'qualify_info': 'आप {count} सरकारी स्वास्थ्य कार्यक्रमों के लिए योग्य हैं। अपनी कवरेज को अधिकतम करने और जेब से खर्च कम करने के लिए आवेदन करने पर विचार करें।',
        'accident_estimation': 'दुर्घटना/चोट लागत अनुमान',
        'accident_description': '''दुर्घटनाओं या चोटों के लिए अतिरिक्त बीमा लागत का अनुमान लगाएं। यह आपको संभावित 
जेब से खर्च को समझने और अप्रत्याशित चिकित्सा घटनाओं के लिए योजना बनाने में मदद करता है।''',
        'your_profile': 'आपकी प्रोफ़ाइल',
        'accident_details': 'दुर्घटना/चोट विवरण',
        'accident_type': 'दुर्घटना/चोट का प्रकार',
        'accident_type_help': 'दुर्घटना या चोट का प्रकार चुनें',
        'car_accident': 'कार दुर्घटना',
        'fall': 'गिरना',
        'sports_injury': 'खेल चोट',
        'workplace_injury': 'कार्यस्थल चोट',
        'other': 'अन्य',
        'severity': 'गंभीरता स्तर',
        'severity_help': 'मामूली: कट, चोट | मध्यम: मोच, मामूली फ्रैक्चर | गंभीर: बड़े फ्रैक्चर, आंतरिक चोटें | क्रिटिकल: जीवन-धमकी',
        'minor': 'मामूली',
        'moderate': 'मध्यम',
        'severe': 'गंभीर',
        'critical': 'क्रिटिकल',
        'recovery_time': 'अनुमानित पुनर्प्राप्ति समय (दिन)',
        'recovery_help': 'पूर्ण पुनर्प्राप्ति के लिए आवश्यक दिनों की संख्या',
        'hospitalization': 'अस्पताल में भर्ती आवश्यक?',
        'hospitalization_help': 'क्या आपको अस्पताल में रहने की आवश्यकता होगी?',
        'surgery': 'सर्जरी आवश्यक?',
        'surgery_help': 'क्या शल्य चिकित्सा हस्तक्षेप आवश्यक होगा?',
        'recovery_period': 'पुनर्प्राप्ति अवधि',
        'days': 'दिन',
        'months': 'महीने',
        'estimate_accident_button': '💉 दुर्घटना/चोट लागत का अनुमान लगाएं',
        'cost_estimation_results': 'लागत अनुमान परिणाम',
        'base_annual_insurance': 'आधार वार्षिक बीमा',
        'accident_injury_cost': 'दुर्घटना/चोट लागत',
        'total_cost': 'कुल लागत',
        'cost_increase': 'लागत वृद्धि',
        'cost_breakdown': 'लागत विवरण',
        'detailed_breakdown': 'विस्तृत लागत विवरण',
        'component': 'घटक',
        'financial_planning': '💰 वित्तीय योजना',
        'immediate_costs': 'तत्काल लागत',
        'emergency_treatment': 'आपातकालीन उपचार',
        'hospital_stay': 'अस्पताल में रहना',
        'surgery_cost': 'सर्जरी',
        'ongoing_costs': 'चल रही लागत',
        'daily_care': 'दैनिक देखभाल',
        'recovery_medication': 'पुनर्प्राप्ति और दवा',
        'monthly_average': 'मासिक औसत',
        'tab_trends': '📈 लागत रुझान',
        'tab_chatbot': '🤖 AI चैटबॉट',
        'tab_document': '📄 दस्तावेज़ विश्लेषक',
        'tab_quotes': '💵 रीयल-टाइम कोट्स',
        'tab_tax': '🧾 कर लाभ',
        'cost_trends_dashboard': 'लागत रुझान डैशबोर्ड',
        'trends_description': 'अपने भविष्यवाणी इतिहास के आधार पर समय के साथ अपनी अनुमानित बीमा लागत कैसे बदलती है, इसे ट्रैक करें',
        'trend_over_time': 'समय के साथ लागत रुझान',
        'predictions_count': 'कुल भविष्यवाणियाँ',
        'avg_predicted_cost': 'औसत अनुमानित लागत',
        'cost_range': 'लागत सीमा',
        'highest_cost': 'उच्चतम लागत',
        'lowest_cost': 'न्यूनतम लागत',
        'cost_by_age_group': 'आयु समूह द्वारा औसत लागत',
        'cost_by_smoker': 'धूम्रपान स्थिति द्वारा लागत वितरण',
        'no_trends_data': 'कोई भविष्यवाणी इतिहास उपलब्ध नहीं है। रुझान देखने के लिए कुछ भविष्यवाणियाँ करें!',
        'ai_chatbot': 'AI बीमा सलाहकार',
        'chatbot_description': 'स्वास्थ्य बीमा, कवरेज, प्रीमियम के बारे में कुछ भी पूछें या व्यक्तिगत सलाह प्राप्त करें',
        'ask_question': 'अपना प्रश्न पूछें',
        'send': 'भेजें',
        'chatbot_placeholder': 'उदाहरण: स्वास्थ्य और जीवन बीमा में क्या अंतर है?',
        'chatbot_thinking': 'सोच रहा है...',
        'setup_ai': 'AI इंटीग्रेशन सेटअप करें',
        'ai_not_configured': 'AI चैटबॉट कॉन्फ़िगर नहीं है। इस सुविधा का उपयोग करने के लिए कृपया OpenAI इंटीग्रेशन सेटअप करें।',
        'document_analyzer': 'बीमा पॉलिसी दस्तावेज़ विश्लेषक',
        'doc_description': 'AI-संचालित विश्लेषण और अंतर्दृष्टि प्राप्त करने के लिए अपनी बीमा पॉलिसी PDF अपलोड करें',
        'upload_policy': 'बीमा पॉलिसी अपलोड करें (PDF)',
        'analyze_button': '🔍 दस्तावेज़ का विश्लेषण करें',
        'analyzing': 'दस्तावेज़ का विश्लेषण कर रहे हैं...',
        'analysis_results': 'विश्लेषण परिणाम',
        'key_points': 'मुख्य बिंदु',
        'coverage_details': 'कवरेज विवरण',
        'exclusions': 'बहिष्करण',
        'premium_info': 'प्रीमियम जानकारी',
        'no_document': 'कृपया विश्लेषण के लिए एक PDF दस्तावेज़ अपलोड करें',
        'realtime_quotes': 'रीयल-टाइम बीमा कोट्स',
        'quotes_description': 'अपनी प्रोफ़ाइल के आधार पर कई प्रदाताओं से तत्काल बीमा कोट्स प्राप्त करें',
        'get_quotes': '💰 बीमा कोट्स प्राप्त करें',
        'fetching_quotes': 'प्रदाताओं से कोट्स प्राप्त कर रहे हैं...',
        'available_plans': 'उपलब्ध बीमा योजनाएं',
        'provider': 'प्रदाता',
        'plan_name': 'योजना का नाम',
        'annual_premium': 'वार्षिक प्रीमियम',
        'coverage_amount': 'कवरेज राशि',
        'key_features': 'मुख्य विशेषताएं',
        'compare_plans': 'योजनाओं की तुलना करें',
        'quotes_disclaimer': 'नोट: ये अनुमानित कोट्स हैं। वास्तविक प्रीमियम चिकित्सा अंडरराइटिंग के आधार पर भिन्न हो सकते हैं।',
        'tax_calculator': 'बीमा कर लाभ कैलकुलेटर',
        'tax_description': 'आयकर अधिनियम की धारा 80D के तहत कर कटौती की गणना करें',
        'premium_paid': 'भुगतान किया गया वार्षिक प्रीमियम (₹)',
        'age_category': 'आयु श्रेणी',
        'below_60': '60 वर्ष से कम',
        'above_60': '60 वर्ष से अधिक (वरिष्ठ नागरिक)',
        'parents_premium': 'माता-पिता का प्रीमियम भुगतान (₹)',
        'parents_age': 'माता-पिता की आयु श्रेणी',
        'preventive_checkup': 'निवारक स्वास्थ्य जांच लागत (₹)',
        'calculate_tax': '🧾 कर लाभ की गणना करें',
        'tax_benefit_results': 'कर लाभ सारांश',
        'self_deduction': 'स्व/परिवार कटौती',
        'parents_deduction': 'माता-पिता कटौती',
        'checkup_deduction': 'स्वास्थ्य जांच कटौती',
        'total_deduction': 'कुल कटौती (80D)',
        'tax_saved_30': 'बचाया गया कर (30% ब्रैकेट)',
        'tax_saved_20': 'बचाया गया कर (20% ब्रैकेट)',
        'tax_saved_10': 'बचाया गया कर (10% ब्रैकेट)',
        'section_80d_info': '💡 धारा 80D जानकारी',
        'deduction_limits': 'कटौती सीमाएं',
        'self_limit': 'स्व/पति-पत्नी/बच्चे: ₹25,000 (₹50,000 यदि वरिष्ठ नागरिक)',
        'parents_limit': 'माता-पिता: ₹25,000 (₹50,000 यदि वरिष्ठ नागरिक)',
        'checkup_limit': 'निवारक स्वास्थ्य जांच: ₹5,000 (उपरोक्त सीमाओं में शामिल)',
        'max_deduction': 'अधिकतम कुल कटौती: ₹1,00,000',
        'login': 'लॉगिन',
        'signup': 'साइन अप',
        'username': 'उपयोगकर्ता नाम',
        'password': 'पासवर्ड',
        'logout': 'लॉगआउट',
        'auth_welcome': 'CareCost Predictor में आपका स्वागत है',
        'auth_error': 'अमान्य उपयोगकर्ता नाम या पासवर्ड',
        'auth_success': 'सफलतापूर्वक लॉगिन किया गया!',
        'no_account': 'खाता नहीं है?',
        'have_account': 'पहले से ही एक खाता है?',
        'create_account': 'खाता बनाएं',
        'email': 'ईमेल आईडी',
        'confirm_password': 'पासवर्ड की पुष्टि करें',
        'passwords_dont_match': 'पासवर्ड मेल नहीं खाते',
        'tab_admin': '🔑 एडमिन डैशबोर्ड',
        'admin_title': 'प्रशासनिक डैशबोर्ड',
        'registered_users': '👥 पंजीकृत उपयोगकर्ता',
        'username': 'उपयोगकर्ता नाम',
        'email_label': 'ईमेल आईडी',
        'tab_receipt_analyzer': '🧾 रसीद विश्लेषक',
        'receipt_analyzer_title': 'चिकित्सा रसीद और विवरण विश्लेषक',
        'receipt_analyzer_desc': 'दवाओं, निर्देशों और महत्वपूर्ण चिकित्सा विवरणों को निकालने के लिए अपनी चिकित्सा रसीदें या डॉक्टर के पर्चे (छवि या PDF) अपलोड करें।',
        'upload_receipt': 'रसीद या पर्चा अपलोड करें',
        'analyze_receipt_button': '🔍 रसीद का विश्लेषण करें',
        'extracted_medicines': '💊 निकाली गई दवाएं',
        'doctor_instructions': '👨‍⚕️ डॉक्टर के निर्देश',
        'important_details': 'ℹ️ महत्वपूर्ण विवरण'
    },
    'ta': {
        'page_title': 'மருத்துவ காப்பீட்டு செலவு கணிப்பு',
        'main_title': '🏥 மருத்துவ காப்பீட்டு செலவு கணிப்பு',
        'main_description': '''இந்த பயன்பாடு தனிப்பட்ட சுகாதார மற்றும் மக்கள்தொகை காரணிகளின் அடிப்படையில் மருத்துவ காப்பீட்டு செலவுகளை கணிக்க மேம்பட்ட இயந்திர கற்றல் (ரேண்டம் ஃபாரஸ்ட் மற்றும் XGBoost) ஐப் பயன்படுத்துகிறது. பல்வேறு காரணிகள் காப்பீட்டு செலவுகளை எவ்வாறு பாதிக்கின்றன என்பதை ஆராய்ந்து 
அரசு மற்றும் தனியார் காப்பீட்டு விருப்பங்களை ஒப்பிடுங்கள்.''',
        'language_selector': 'மொழி / Language / भाषा',
        'model_info': '📊 மாதிரி தகவல்',
        'model_type': 'மாதிரி வகை',
        'training_accuracy': 'பயிற்சி துல்லியம்',
        'testing_accuracy': 'சோதனை துல்லியம்',
        'dataset_size': 'தரவுத்தொகுப்பு அளவு',
        'samples': 'மாதிரிகள்',
        'model_comparison': '🏆 மாதிரி ஒப்பீடு',
        'random_forest': 'ரேண்டம் ஃபாரஸ்ட்',
        'xgboost': 'XGBoost',
        'xgb_improved': 'XGBoost மேம்பாடு',
        'rf_better': 'ரேண்டம் ஃபாரஸ்ட் சிறந்த செயல்திறன்',
        'models_equal': 'இரண்டு மாதிரிகளும் சமமாக செயல்பட்டன',
        'dataset_stats': '📈 தரவுத்தொகுப்பு புள்ளிவிவரங்கள்',
        'age_range': 'வயது வரம்பு',
        'years': 'ஆண்டுகள்',
        'bmi_range': 'BMI வரம்பு',
        'avg_cost': 'சராசரி செலவு',
        'smokers': 'புகைபிடிப்பவர்கள்',
        'export_data': '📥 தரவு ஏற்றுமதி',
        'predictions_made': 'கணிப்புகள் செய்யப்பட்டன',
        'download_csv': '📊 CSV பதிவிறக்கம்',
        'clear_history': '🗑️ வரலாற்றை அழி',
        'no_predictions': 'இன்னும் கணிப்புகள் இல்லை',
        'tab_prediction': '🔮 கணிப்பு',
        'tab_visualizations': '📊 காட்சிப்படுத்தல்கள்',
        'tab_whatif': '🔄 என்றால் என்ன பகுப்பாய்வு',
        'tab_cost_comparison': '💰 செலவு ஒப்பீடு',
        'tab_accident': '🚑 விபத்து/காயம் செலவு',
        'insurance_cost_prediction': 'காப்பீட்டு செலவு கணிப்பு',
        'personal_info': 'தனிப்பட்ட தகவல்',
        'age': 'வயது',
        'age_help': 'உங்கள் தற்போதைய வயது ஆண்டுகளில்',
        'gender': 'பாலினம்',
        'male': 'ஆண்',
        'female': 'பெண்',
        'children': 'குழந்தைகளின் எண்ணிக்கை',
        'region': 'பகுதி',
        'northeast': 'வடகிழக்கு',
        'northwest': 'வடமேற்கு',
        'southeast': 'தென்கிழக்கு',
        'southwest': 'தென்மேற்கு',
        'health_info': 'சுகாதார தகவல்',
        'bmi': 'BMI (உடல் நிறை குறியீடு)',
        'bmi_help': 'BMI = எடை(கிலோ) / உயரம்(மீ)²',
        'smoking_status': 'புகைபிடித்தல் நிலை',
        'yes': 'ஆம்',
        'no': 'இல்லை',
        'bmi_category': 'BMI வகை',
        'underweight': 'குறைவான எடை',
        'normal_weight': 'சாதாரண எடை',
        'overweight': 'அதிக எடை',
        'obese': 'பருமன்',
        'predict_button': '🔮 காப்பீட்டு செலவை கணிக்கவும்',
        'prediction_results': 'கணிப்பு முடிவுகள்',
        'predicted_annual_cost': 'கணிக்கப்பட்ட வருடாந்திர செலவு',
        'risk_level': 'ஆபத்து நிலை',
        'monthly_premium': 'மாதாந்திர பிரீமியம் (மதிப்பீடு)',
        'cost_factor_analysis': 'செலவு காரணி பகுப்பாய்வு',
        'age_factor': 'வயது காரணி',
        'bmi_factor': 'BMI காரணி',
        'smoking_factor': 'புகைபிடித்தல் காரணி',
        'children_factor': 'குழந்தைகள் காரணி',
        'factor_impact_title': 'ஒவ்வொரு காரணியின் செலவில் மதிப்பிடப்பட்ட தாக்கம்',
        'export_report': 'அறிக்கை ஏற்றுமதி',
        'download_pdf': '📄 PDF அறிக்கை பதிவிறக்கம்',
        'interactive_visualizations': 'ஊடாடும் தரவு காட்சிப்படுத்தல்கள்',
        'cost_vs_age': 'காப்பீட்டு செலவு vs வயது',
        'insurance_cost': 'காப்பீட்டு செலவு (₹)',
        'age_years': 'வயது (ஆண்டுகள்)',
        'avg_cost_children': 'குழந்தைகளின் எண்ணிக்கையின்படி சராசரி காப்பீட்டு செலவு',
        'average_cost': 'சராசரி செலவு (₹)',
        'number_of_children': 'குழந்தைகளின் எண்ணிக்கை',
        'cost_vs_bmi': 'காப்பீட்டு செலவு vs BMI',
        'smoking_impact': 'காப்பீட்டு செலவில் புகைபிடித்தல் நிலையின் தாக்கம்',
        'smoker': 'புகைபிடிப்பவர்',
        'regional_cost_analysis': 'பிராந்திய செலவு பகுப்பாய்வு',
        'average': 'சராசரி',
        'minimum': 'குறைந்தபட்சம்',
        'maximum': 'அதிகபட்சம்',
        'whatif_tool': 'என்றால் என்ன பகுப்பாய்வு கருவி',
        'whatif_description': 'மாற்றங்கள் உங்கள் காப்பீட்டு செலவு கணிப்பை எவ்வாறு பாதிக்கின்றன என்பதைக் காண கீழே உள்ள அளவுருக்களை சரிசெய்யவும்',
        'baseline_scenario': '📍 அடிப்படை சூழ்நிலை',
        'baseline_age': 'அடிப்படை வயது',
        'baseline_gender': 'அடிப்படை பாலினம்',
        'baseline_bmi': 'அடிப்படை BMI',
        'baseline_children': 'அடிப்படை குழந்தைகள்',
        'baseline_smoker': 'அடிப்படை புகைபிடித்தல்',
        'baseline_region': 'அடிப்படை பகுதி',
        'baseline_cost': 'அடிப்படை செலவு',
        'whatif_scenario': '🔄 என்றால் என்ன சூழ்நிலை',
        'whatif_age': 'என்றால் என்ன வயது',
        'whatif_gender': 'என்றால் என்ன பாலினம்',
        'whatif_bmi': 'என்றால் என்ன BMI',
        'whatif_children': 'என்றால் என்ன குழந்தைகள்',
        'whatif_smoker': 'என்றால் என்ன புகைபிடித்தல்',
        'whatif_region': 'என்றால் என்ன பகுதி',
        'whatif_cost': 'என்றால் என்ன செலவு',
        'scenario_comparison': 'சூழ்நிலை ஒப்பீடு',
        'comparison_title': 'செலவு ஒப்பீடு: அடிப்படை vs என்றால் என்ன',
        'baseline': 'அடிப்படை',
        'whatif': 'என்றால் என்ன',
        'parameter_changes': 'அளவுரு மாற்றங்கள்',
        'no_changes': 'எந்த அளவுருவும் மாற்றப்படவில்லை. தாக்கத்தைக் காண மதிப்புகளை சரிசெய்யவும்.',
        'govt_vs_private': 'அரசு vs தனியார் காப்பீட்டு ஒப்பீடு',
        'govt_vs_private_desc': '''அரசு சுகாதார திட்டங்கள் மற்றும் தனியார் காப்பீட்டு விருப்பங்களுக்கு இடையே மதிப்பிடப்பட்ட செலவுகளை ஒப்பிடுங்கள்.
அரசு திட்டங்கள் பொதுவாக குறைந்த பிரீமியங்களுடன் அடிப்படை கவரேஜை வழங்குகின்றன, அதே சமயம் தனியார் காப்பீடு 
அதிக பிரீமியங்களுடன் விரிவான கவரேஜை வழங்குகிறது.''',
        'enter_details': 'உங்கள் விவரங்களை உள்ளிடவும்',
        'compare_button': '💰 காப்பீட்டு விருப்பங்களை ஒப்பிடுங்கள்',
        'comparison_results': 'செலவு ஒப்பீட்டு முடிவுகள்',
        'govt_scheme': '🏛️ அரசு திட்டம்',
        'govt_coverage': 'அரசு கவரேஜ்',
        'out_of_pocket': 'உங்கள் சொந்த செலவு',
        'coverage_percentage': 'கவரேஜ் சதவீதம்',
        'pros': 'நன்மைகள்:',
        'cons': 'குறைபாடுகள்:',
        'govt_pro1': 'குறைந்த பிரீமியங்கள்',
        'govt_pro2': 'அடிப்படை கவரேஜ் சேர்க்கப்பட்டுள்ளது',
        'govt_pro3': 'அரசு மானியம்',
        'govt_con1': 'வரையறுக்கப்பட்ட கவரேஜ்',
        'govt_con2': 'அதிக சொந்த செலவுகள்',
        'govt_con3': 'குறைவான மருத்துவமனை தேர்வுகள்',
        'private_insurance': '🏥 தனியார் காப்பீடு',
        'base_plan_cost': 'அடிப்படை திட்ட செலவு',
        'premium_plan_cost': 'பிரீமியம் திட்ட செலவு',
        'private_pro1': 'விரிவான கவரேஜ்',
        'private_pro2': 'பரந்த மருத்துவமனை நெட்வொர்க்',
        'private_pro3': 'கூடுதல் நன்மைகள்',
        'private_con1': 'அதிக பிரீமியங்கள்',
        'private_con2': 'சிக்கலான விதிமுறைகள்',
        'private_con3': 'காத்திருப்பு காலங்கள்',
        'visual_breakdown': 'காட்சி செலவு விவரம்',
        'insurance_comparison': 'காப்பீட்டு செலவு ஒப்பீடு',
        'govt_coverage_label': 'அரசு\nகவரேஜ்',
        'govt_oop_label': 'அரசு\nசொந்த செலவு',
        'private_base_label': 'தனியார்\nஅடிப்படை திட்டம்',
        'private_premium_label': 'தனியார்\nபிரீமியம் திட்டம்',
        'government': 'அரசு',
        'private': 'தனியார்',
        'predicted_total': 'கணிக்கப்பட்ட மொத்த செலவு',
        'recommendations': '💡 பரிந்துரைகள்',
        'govt_economical': '✅ நீங்கள் சொந்த செலவுகளை நிர்வகிக்க முடிந்தால் அரசு திட்டம் மிகவும் சிக்கனமாக இருக்கலாம்.',
        'private_better': 'ℹ️ தனியார் காப்பீடு விரிவான கவரேஜுடன் சிறந்த மதிப்பை வழங்கலாம்.',
        'smoker_warning': '⚠️ புகைபிடிப்பவராக, காப்பீட்டு செலவுகளை கணிசமாக குறைக்க புகைபிடிப்பதை நிறுத்துவதைக் கருத்தில் கொள்ளுங்கள். புகைபிடித்தல் செலவுகளை 150-250% அதிகரிக்கலாம்.',
        'bmi_warning': '⚠️ அதிக BMI காப்பீட்டு செலவுகளை அதிகரிக்கிறது. பிரீமியங்களை குறைக்க எடை மேலாண்மை திட்டத்தை கருத்தில் கொள்ளுங்கள்.',
        'eligible_schemes': '🏛️ தகுதியான அரசு சுகாதார திட்டங்கள்',
        'schemes_description': 'உங்கள் சுயவிவரத்தின் அடிப்படையில், நீங்கள் பின்வரும் அரசு உதவி திட்டங்களுக்கு தகுதியுடையவராக இருக்கலாம்:',
        'high_priority': 'உயர்',
        'medium_priority': 'நடுத்தர',
        'priority': 'முன்னுரிமை',
        'eligibility': 'தகுதி',
        'coverage': 'கவரேஜ்',
        'benefits': 'நன்மைகள்',
        'how_to_apply': 'எவ்வாறு விண்ணப்பிப்பது',
        'highly_recommended': '✅ இந்த திட்டம் உங்கள் சுயவிவரத்திற்கு மிகவும் பரிந்துரைக்கப்படுகிறது',
        'qualify_info': 'நீங்கள் {count} அரசு சுகாதார திட்டங்களுக்கு தகுதி பெறுகிறீர்கள். உங்கள் கவரேஜை அதிகரிக்கவும் சொந்த செலவுகளை குறைக்கவும் விண்ணப்பிப்பதை கருத்தில் கொள்ளுங்கள்.',
        'accident_estimation': 'விபத்து/காயம் செலவு மதிப்பீடு',
        'accident_description': '''விபத்துகள் அல்லது காயங்களுக்கான கூடுதல் காப்பீட்டு செலவுகளை மதிப்பிடுங்கள். இது உங்களுக்கு சாத்தியமான 
சொந்த செலவுகளை புரிந்துகொள்ளவும் எதிர்பாராத மருத்துவ நிகழ்வுகளுக்கு திட்டமிடவும் உதவுகிறது.''',
        'your_profile': 'உங்கள் சுயவிவரம்',
        'accident_details': 'விபத்து/காயம் விவரங்கள்',
        'accident_type': 'விபத்து/காயத்தின் வகை',
        'accident_type_help': 'விபத்து அல்லது காயத்தின் வகையை தேர்ந்தெடுக்கவும்',
        'car_accident': 'கார் விபத்து',
        'fall': 'விழுதல்',
        'sports_injury': 'விளையாட்டு காயம்',
        'workplace_injury': 'பணியிட காயம்',
        'other': 'மற்றவை',
        'severity': 'தீவிரத்தன்மை நிலை',
        'severity_help': 'சிறிய: வெட்டுக்கள், காயங்கள் | நடுத்தர: சுளுக்குகள், சிறிய எலும்பு முறிவுகள் | கடுமையான: பெரிய எலும்பு முறிவுகள், உள் காயங்கள் | முக்கியமான: உயிருக்கு ஆபத்தானது',
        'minor': 'சிறிய',
        'moderate': 'நடுத்தர',
        'severe': 'கடுமையான',
        'critical': 'முக்கியமான',
        'recovery_time': 'மதிப்பிடப்பட்ட மீட்பு நேரம் (நாட்கள்)',
        'recovery_help': 'முழு மீட்புக்கு தேவையான நாட்களின் எண்ணிக்கை',
        'hospitalization': 'மருத்துவமனையில் சேர்க்கை தேவையா?',
        'hospitalization_help': 'நீங்கள் மருத்துவமனையில் தங்க வேண்டுமா?',
        'surgery': 'அறுவை சிகிச்சை தேவையா?',
        'surgery_help': 'அறுவை சிகிச்சை தலையீடு அவசியமா?',
        'recovery_period': 'மீட்பு காலம்',
        'days': 'நாட்கள்',
        'months': 'மாதங்கள்',
        'estimate_accident_button': '💉 விபத்து/காயம் செலவை மதிப்பிடுங்கள்',
        'cost_estimation_results': 'செலவு மதிப்பீட்டு முடிவுகள்',
        'base_annual_insurance': 'அடிப்படை வருடாந்திர காப்பீடு',
        'accident_injury_cost': 'விபத்து/காயம் செலவு',
        'total_cost': 'மொத்த செலவு',
        'cost_increase': 'செலவு அதிகரிப்பு',
        'cost_breakdown': 'செலவு விவரம்',
        'detailed_breakdown': 'விரிவான செலவு விவரம்',
        'component': 'கூறு',
        'financial_planning': '💰 நிதி திட்டமிடல்',
        'immediate_costs': 'உடனடி செலவுகள்',
        'emergency_treatment': 'அவசர சிகிச்சை',
        'hospital_stay': 'மருத்துவமனையில் தங்குதல்',
        'surgery_cost': 'அறுவை சிகிச்சை',
        'ongoing_costs': 'நடந்துகொண்டிருக்கும் செலவுகள்',
        'daily_care': 'தினசரி பராமரிப்பு',
        'recovery_medication': 'மீட்பு மற்றும் மருந்து',
        'monthly_average': 'மாதாந்திர சராசரி',
        'tab_trends': '📈 செலவு போக்குகள்',
        'tab_chatbot': '🤖 AI சாட்பாட்',
        'tab_document': '📄 ஆவண பகுப்பாய்வி',
        'tab_quotes': '💵 நேரடி மேற்கோள்கள்',
        'tab_tax': '🧾 வரி நன்மைகள்',
        'cost_trends_dashboard': 'செலவு போக்குகள் டாஷ்போர்டு',
        'trends_description': 'உங்கள் கணிப்பு வரலாற்றின் அடிப்படையில் காலப்போக்கில் உங்கள் கணிக்கப்பட்ட காப்பீட்டு செலவுகள் எவ்வாறு மாறுகின்றன என்பதைக் கண்காணிக்கவும்',
        'trend_over_time': 'காலப்போக்கில் செலவு போக்கு',
        'predictions_count': 'மொத்த கணிப்புகள்',
        'avg_predicted_cost': 'சராசரி கணிக்கப்பட்ட செலவு',
        'cost_range': 'செலவு வரம்பு',
        'highest_cost': 'அதிகபட்ச செலவு',
        'lowest_cost': 'குறைந்தபட்ச செலவு',
        'cost_by_age_group': 'வயது குழுவின்படி சராசரி செலவு',
        'cost_by_smoker': 'புகைபிடித்தல் நிலையின்படி செலவு விநியோகம்',
        'no_trends_data': 'கணிப்பு வரலாறு கிடைக்கவில்லை. போக்குகளைக் காண சில கணிப்புகளை செய்யுங்கள்!',
        'ai_chatbot': 'AI காப்பீட்டு ஆலோசகர்',
        'chatbot_description': 'சுகாதார காப்பீடு, கவரேஜ், பிரீமியங்கள் அல்லது தனிப்பயனாக்கப்பட்ட ஆலோசனை பற்றி எதையும் கேளுங்கள்',
        'ask_question': 'உங்கள் கேள்வியைக் கேளுங்கள்',
        'send': 'அனுப்பு',
        'chatbot_placeholder': 'உதாரணம்: சுகாதார மற்றும் ஆயுள் காப்பீட்டுக்கு இடையே என்ன வித்தியாசம்?',
        'chatbot_thinking': 'சிந்தித்துக்கொண்டிருக்கிறது...',
        'setup_ai': 'AI ஒருங்கிணைப்பை அமைக்கவும்',
        'ai_not_configured': 'AI சாட்பாட் கட்டமைக்கப்படவில்லை. இந்த அம்சத்தைப் பயன்படுத்த OpenAI ஒருங்கிணைப்பை அமைக்கவும்.',
        'document_analyzer': 'காப்பீட்டு பாலிசி ஆவண பகுப்பாய்வி',
        'doc_description': 'AI-இயங்கும் பகுப்பாய்வு மற்றும் நுண்ணறிவுகளைப் பெற உங்கள் காப்பீட்டு பாலிசி PDF ஐ பதிவேற்றவும்',
        'upload_policy': 'காப்பீட்டு பாலிசியை பதிவேற்றவும் (PDF)',
        'analyze_button': '🔍 ஆவணத்தை பகுப்பாய்வு செய்யவும்',
        'analyzing': 'ஆவணத்தை பகுப்பாய்வு செய்கிறது...',
        'analysis_results': 'பகுப்பாய்வு முடிவுகள்',
        'key_points': 'முக்கிய புள்ளிகள்',
        'coverage_details': 'கவரேஜ் விவரங்கள்',
        'exclusions': 'விலக்குகள்',
        'premium_info': 'பிரீமியம் தகவல்',
        'no_document': 'பகுப்பாய்வு செய்ய PDF ஆவணத்தைப் பதிவேற்றவும்',
        'realtime_quotes': 'நேரடி காப்பீட்டு மேற்கோள்கள்',
        'quotes_description': 'உங்கள் சுயவிவரத்தின் அடிப்படையில் பல வழங்குநர்களிடமிருந்து உடனடி காப்பீட்டு மேற்கோள்களைப் பெறுங்கள்',
        'get_quotes': '💰 காப்பீட்டு மேற்கோள்களைப் பெறுங்கள்',
        'fetching_quotes': 'வழங்குநர்களிடமிருந்து மேற்கோள்களைப் பெறுகிறது...',
        'available_plans': 'கிடைக்கக்கூடிய காப்பீட்டு திட்டங்கள்',
        'provider': 'வழங்குநர்',
        'plan_name': 'திட்டத்தின் பெயர்',
        'annual_premium': 'வருடாந்திர பிரீமியம்',
        'coverage_amount': 'கவரேஜ் தொகை',
        'key_features': 'முக்கிய அம்சங்கள்',
        'compare_plans': 'திட்டங்களை ஒப்பிடுங்கள்',
        'quotes_disclaimer': 'குறிப்பு: இவை மதிப்பிடப்பட்ட மேற்கோள்கள். உண்மையான பிரீமியங்கள் மருத்துவ அண்டர்ரைட்டிங்கின் அடிப்படையில் மாறுபடலாம்.',
        'tax_calculator': 'காப்பீட்டு வரி நன்மை கால்குலேட்டர்',
        'tax_description': 'வருமான வரி சட்டத்தின் பிரிவு 80D இன் கீழ் வரி விலக்குகளை கணக்கிடுங்கள்',
        'premium_paid': 'செலுத்தப்பட்ட வருடாந்திர பிரீமியம் (₹)',
        'age_category': 'வயது வகை',
        'below_60': '60 வயதுக்குக் குறைவானவர்',
        'above_60': '60 வயதுக்கு மேற்பட்டவர் (மூத்த குடிமகன்)',
        'parents_premium': 'பெற்றோர் பிரீமியம் செலுத்தப்பட்டது (₹)',
        'parents_age': 'பெற்றோர் வயது வகை',
        'preventive_checkup': 'தடுப்பு சுகாதார பரிசோதனை செலவு (₹)',
        'calculate_tax': '🧾 வரி நன்மையை கணக்கிடுங்கள்',
        'tax_benefit_results': 'வரி நன்மை சுருக்கம்',
        'self_deduction': 'சுய/குடும்ப விலக்கு',
        'parents_deduction': 'பெற்றோர் விலக்கு',
        'checkup_deduction': 'சுகாதார பரிசோதனை விலக்கு',
        'total_deduction': 'மொத்த விலக்கு (80D)',
        'tax_saved_30': 'சேமிக்கப்பட்ட வரி (30% வரம்பு)',
        'tax_saved_20': 'சேமிக்கப்பட்ட வரி (20% வரம்பு)',
        'tax_saved_10': 'சேமிக்கப்பட்ட வரி (10% வரம்பு)',
        'section_80d_info': '💡 பிரிவு 80D தகவல்',
        'deduction_limits': 'விலக்கு வரம்புகள்',
        'self_limit': 'சுய/மனைவி/குழந்தைகள்: ₹25,000 (₹50,000 மூத்த குடிமகன் என்றால்)',
        'parents_limit': 'பெற்றோர்: ₹25,000 (₹50,000 மூத்த குடிமகன் என்றால்)',
        'checkup_limit': 'தடுப்பு சுகாதார பரிசோதனை: ₹5,000 (மேலே உள்ள வரம்புகளில் சேர்க்கப்பட்டுள்ளது)',
        'max_deduction': 'அதிகபட்ச மொத்த விலக்கு: ₹1,00,000',
        'tab_receipt_analyzer': '🧾 ரசீது பகுப்பாய்வி',
        'receipt_analyzer_title': 'மருத்துவ ரசீது மற்றும் விவர பகுப்பாய்வி',
        'receipt_analyzer_desc': 'மருந்துகள், அறிவுறுத்தல்கள் மற்றும் முக்கிய மருத்துவ விவரங்களை எடுக்க உங்கள் மருத்துவ ரசீதுகள் அல்லது மருத்துவர் பரிந்துரைகளை (படம் அல்லது PDF) பதிவேற்றவும்.',
        'upload_receipt': 'ரசீது அல்லது பரிந்துரையை பதிவேற்றவும்',
        'analyze_receipt_button': '🔍 ரசீதை பகுப்பாய்வு செய்யவும்',
        'extracted_medicines': '💊 எடுக்கப்பட்ட மருந்துகள்',
        'doctor_instructions': '👨‍⚕️ மருத்துவரின் அறிவுறுத்தல்கள்',
        'important_details': 'ℹ️ முக்கிய விவரங்கள்',
        'login': 'உள்நுழை',
        'signup': 'பதிவு செய்க',
        'username': 'பயனர் பெயர்',
        'password': 'கடவுச்சொல்',
        'logout': 'வெளியேறு',
        'auth_welcome': 'கையர் காஸ்ட் பிரிடிக்டருக்கு உங்களை வரவேற்கிறோம்',
        'auth_error': 'தவறான பயனர் பெயர் அல்லது கடவுச்சொல்',
        'auth_success': 'வெற்றிகரமாக உள்நுழைந்தீர்கள்!',
        'no_account': 'கணக்கு இல்லையா?',
        'have_account': 'ஏற்கனவே கணக்கு உள்ளதா?',
        'create_account': 'கணக்கை உருவாக்கு',
        'email': 'மின்னஞ்சல் ஐடி',
        'confirm_password': 'கடவுச்சொல்லை உறுதிப்படுத்தவும்',
        'passwords_dont_match': 'கடவுச்சொற்கள் பொருந்தவில்லை',
        'tab_admin': '🔑 நிர்வாக டாஷ்போர்டு',
        'admin_title': 'நிர்வாக டாஷ்போர்டு',
        'registered_users': '👥 பதிவு செய்யப்பட்ட பயனர்கள்',
        'username': 'பயனர் பெயர்',
        'email_label': 'மின்னஞ்சல் ஐடி'
    }
}

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide"
)

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
                           color_continuous_scale='RdYlGn_r')
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
                            color_discrete_map={'yes': '#ff4444', 'no': '#44ff44'},
                            trendline='lowess')
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Cost vs Children
        avg_by_children = df.groupby('children')['charges'].mean().reset_index()
        fig_children = px.bar(avg_by_children, x='children', y='charges',
                             title=t('avg_cost_children'),
                             labels={'charges': t('average_cost'), 'children': t('number_of_children')})
        fig_children.update_layout(height=400)
        st.plotly_chart(fig_children, use_container_width=True)
    
    with viz_col2:
        # Cost vs BMI
        fig_bmi = px.scatter(df, x='bmi', y='charges', color='smoker',
                            title=t('cost_vs_bmi'),
                            labels={'charges': t('insurance_cost'), 'bmi': 'BMI'},
                            color_discrete_map={'yes': '#ff4444', 'no': '#44ff44'},
                            trendline='lowess')
        fig_bmi.update_layout(height=400)
        st.plotly_chart(fig_bmi, use_container_width=True)
        
        # Smoking Impact
        avg_by_smoker = df.groupby('smoker')['charges'].mean().reset_index()
        fig_smoker = px.bar(avg_by_smoker, x='smoker', y='charges',
                           title=t('smoking_impact'),
                           labels={'charges': t('average_cost'), 'smoker': t('smoker')},
                           color='smoker',
                           color_discrete_map={'yes': '#ff4444', 'no': '#44ff44'})
        fig_smoker.update_layout(height=400)
        st.plotly_chart(fig_smoker, use_container_width=True)
    
    # Regional analysis
    st.markdown("---")
    regional_stats = df.groupby('region')['charges'].agg(['mean', 'min', 'max']).reset_index()
    regional_stats.columns = [t('region'), t('average') + ' Cost', t('minimum') + ' Cost', t('maximum') + ' Cost']
    
    fig_region = go.Figure()
    fig_region.add_trace(go.Bar(name=t('average'), x=regional_stats[t('region')], y=regional_stats[t('average') + ' Cost']))
    fig_region.add_trace(go.Bar(name=t('minimum'), x=regional_stats[t('region')], y=regional_stats[t('minimum') + ' Cost']))
    fig_region.add_trace(go.Bar(name=t('maximum'), x=regional_stats[t('region')], y=regional_stats[t('maximum') + ' Cost']))
    fig_region.update_layout(title=t('regional_cost_analysis'), barmode='group', height=400)
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
                           color_discrete_map={t('baseline'): '#3498db', t('whatif'): '#e74c3c'})
    fig_comparison.update_layout(height=400)
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
                           labels={'predicted_cost': t('insurance_cost'), 'timestamp': 'Time'})
        fig_trend.update_traces(mode='lines+markers')
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Additional analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(t('cost_by_age_group'))
            history_df['age_group'] = pd.cut(history_df['age'], bins=[0, 30, 40, 50, 65], labels=['18-30', '31-40', '41-50', '51-64'])
            age_group_avg = history_df.groupby('age_group')['predicted_cost'].mean().reset_index()
            fig_age = px.bar(age_group_avg, x='age_group', y='predicted_cost',
                           labels={'predicted_cost': t('average_cost'), 'age_group': t('age')})
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.subheader(t('cost_by_smoker'))
            smoker_dist = history_df.groupby('smoker')['predicted_cost'].mean().reset_index()
            fig_smoker = px.pie(smoker_dist, values='predicted_cost', names='smoker',
                              title=t('cost_by_smoker'))
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
