import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def generate_explanations(model_path):
    print(f"Loading model and data from {model_path}...")
    data = joblib.load(model_path)
    model = data['model']
    model_name = data['model_name']
    feature_names = data['feature_names']
    
    # Reload original dataset for background distributions
    df = pd.read_csv('data/diabetes.csv')
    X = df.drop('Outcome', axis=1)
    
    print(f"Generating explanations for {model_name}...")
    
    if model_name in ['XGBoost', 'Decision Tree']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Summary Plot
        plt.figure(figsize=(10, 6))
        # For Decision Tree, shap_values is often (N, P, 2) - we want the positive class (1)
        if isinstance(shap_values, list): # Older format or specific models
            vals = shap_values[1]
        elif len(shap_values.shape) == 3: # (samples, features, classes)
            vals = shap_values[:, :, 1]
        else:
            vals = shap_values
            
        shap.summary_plot(vals, X, show=False)
        plt.tight_layout()
        plt.savefig('models/shap_summary.png')
        plt.close()
        
        # Waterfall/Force for sample 0
        plt.figure(figsize=(12, 4))
        # Explanation object for the positive class
        explanation = shap.Explanation(
            values=vals[0],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            data=X.iloc[0].values,
            feature_names=feature_names
        )
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig('models/shap_waterfall_sample.png')
        plt.close()
        
    elif model_name == 'EBM':
        importances = model.term_importances()
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(importances)
        plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
        plt.title('EBM Global Feature Importance')
        plt.tight_layout()
        plt.savefig('models/shap_summary.png') # Using same name for consistency in app
        plt.close()
    
    print("Explanations updated.")

if __name__ == "__main__":
    generate_explanations('models/best_model.pkl')
