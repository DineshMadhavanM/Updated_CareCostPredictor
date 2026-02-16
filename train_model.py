"""
Script to train the machine learning model using existing dataset
Loads insurance_data.csv and trains the model
"""
from utils import train_model
import pandas as pd
import os

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists('insurance_data.csv'):
        print("Error: insurance_data.csv not found. Please provide a dataset file.")
        exit(1)
    
    print("Loading insurance dataset from insurance_data.csv...")
    df = pd.read_csv('insurance_data.csv')
    
    print(f"Dataset loaded with {len(df)} samples")
    print(f"\nDataset Info:")
    print(df.describe())
    
    print("\nTraining machine learning model...")
    model_data = train_model(df)
    
    print(f"\nModel Training Complete!")
    print(f"Model Type: {model_data['model_type']}")
    print(f"Training R² Score: {model_data['train_score']:.4f}")
    print(f"Testing R² Score: {model_data['test_score']:.4f}")
    print(f"Model saved to insurance_model.pkl")
