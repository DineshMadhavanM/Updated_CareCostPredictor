# Medical Insurance Cost Predictor

## Overview

This is a Streamlit-based web application that predicts medical insurance costs using machine learning models (Random Forest and XGBoost). The application analyzes personal health and demographic factors to estimate insurance costs, provides risk assessments, compares government vs private insurance options, and generates detailed PDF reports. It uses a synthetic medical dataset similar to real-world insurance data for training and predictions.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid UI development
- **Visualization**: Plotly (Express and Graph Objects) for interactive charts and data visualization
- **Layout**: Wide layout configuration with sidebar for model information
- **State Management**: Streamlit session state for maintaining prediction history across user interactions

### Backend Architecture
- **ML Models**: Dual model support with Random Forest (primary) and XGBoost (optional)
- **Model Training Pipeline**: 
  - Train/test split for model validation
  - Label encoding for categorical variables (sex, smoker status, region)
  - Model persistence using pickle for deployment
- **Prediction Engine**: 
  - Cost prediction based on age, BMI, smoking status, children, and region
  - Risk level assessment system
  - Government vs private insurance comparison
  - Accident/injury cost estimation with breakdown analysis
- **Data Generation**: Synthetic dataset generator mimicking real-world medical insurance data patterns with realistic correlations:
  - Age factor: Linear cost increase (~$250/year)
  - BMI impact: Exponential for obesity (BMI > 30)
  - Smoking multiplier: 2.5x cost increase
  - Children dependency: ~$500 per child

### Data Storage
- **Local CSV Storage**: Insurance dataset stored in `insurance_data.csv`
- **Model Persistence**: Trained model saved as `insurance_model.pkl` using pickle serialization
- **Session Storage**: In-memory prediction history stored in Streamlit session state
- **Lazy Loading**: Dataset auto-generated on first run if not present

### Key Design Decisions
- **Caching Strategy**: `@st.cache_resource` decorator used to prevent redundant model loading and data initialization
- **Fallback Mechanism**: XGBoost is optional dependency; application gracefully degrades to Random Forest if unavailable
- **Modular Design**: Core ML utilities separated into `utils.py` for reusability and maintainability
- **Reproducibility**: Fixed random seed (42) for consistent synthetic data generation

## External Dependencies

### Python Libraries
- **streamlit**: Web application framework and UI rendering
- **pandas**: Data manipulation and CSV file handling
- **numpy**: Numerical computations and synthetic data generation
- **plotly**: Interactive visualization library (express and graph_objects modules)
- **scikit-learn**: Machine learning framework (RandomForestRegressor, train_test_split, LabelEncoder)
- **xgboost** (optional): Advanced gradient boosting framework
- **pickle**: Model serialization (Python standard library)

### Utility Functions (utils.py)
- `load_model()`: Load persisted ML model
- `predict_cost()`: Generate insurance cost predictions
- `get_risk_level()`: Calculate risk assessment
- `get_govt_vs_private_comparison()`: Compare insurance options
- `generate_pdf_report()`: Export predictions as PDF
- `estimate_accident_injury_cost()`: Calculate accident-related costs
- `get_accident_cost_breakdown()`: Detailed cost analysis
- `get_government_scheme_recom()`: Government scheme recommendations

### Data Files
- `insurance_data.csv`: Training/reference dataset (auto-generated if missing)
- `insurance_model.pkl`: Serialized trained model