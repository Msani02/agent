import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve
)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_and_preprocess_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Specify columns where zero values should be replaced by median
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in cols_to_fix:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
    
    # Handle Class Imbalance with SMOTE
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"Original training shape: {y_train.value_counts().to_dict()}")
    print(f"Resampled training shape: {y_train_res.value_counts().to_dict()}")
    
    return X_train_res, X_test, y_train_res, y_test, X.columns

def train_models(X_train, y_train):
    print("Training models with enhanced RandomizedSearchCV...")
    
    # Scaling for SVC
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    models_config = {
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=RANDOM_STATE),
            'params': {
                'max_depth': [3, 5, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'criterion': ['gini', 'entropy']
            }
        },
        'SVC': {
            'model': SVC(probability=True, random_state=RANDOM_STATE),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']
            },
            'use_scaled': True
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        },
        'EBM': {
            'model': ExplainableBoostingClassifier(random_state=RANDOM_STATE),
            'params': {
                'outer_bags': [8, 16, 24],
                'learning_rate': [0.01, 0.05, 0.1],
                'interactions': [0, 5, 10]
            }
        }
    }
    
    best_estimators = {}
    for name, config in models_config.items():
        print(f"Tuning {name}...")
        X_data = X_train_scaled if config.get('use_scaled') else X_train
        search = RandomizedSearchCV(config['model'], config['params'], n_iter=20, cv=5, scoring='f1', random_state=RANDOM_STATE, n_jobs=-1)
        search.fit(X_data, y_train)
        best_estimators[name] = search.best_estimator_
        print(f"Best params for {name}: {search.best_params_}")

    # Build Ensemble (Voting Classifier)
    print("Building Ensemble Model (Voting Classifier)...")
    # Wrap SVC for consistent scaling in the ensemble
    from sklearn.pipeline import Pipeline
    svc_pipe = Pipeline([('scaler', StandardScaler()), ('svc', best_estimators['SVC'])])
    
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', best_estimators['XGBoost']),
            ('svc', svc_pipe),
            ('ebm', best_estimators['EBM'])
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    best_estimators['Ensemble'] = ensemble
    
    return best_estimators, scaler

def evaluate_models(models, X_test, y_test, scaler):
    results = []
    
    # Single plot for ROC
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        # Handle scaling internally based on model type
        if name == 'SVC':
            X_data = scaler.transform(X_test)
        else:
            X_data = X_test
            
        y_pred = model.predict(X_data)
        y_prob = model.predict_proba(X_data)[:, 1]
        
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        results.append(metrics)
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {metrics['ROC-AUC']:.2f})")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'models/cm_{name.replace(" ", "_").lower()}.png')
        plt.close()

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.savefig('models/roc_comparison.png')
    plt.close()
    
    return pd.DataFrame(results)

def main():
    if not os.path.exists('models'):
        os.makedirs('models')
        
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data('data/diabetes.csv')
    
    best_estimators, scaler = train_models(X_train, y_train)
    
    results_df = evaluate_models(best_estimators, X_test, y_test, scaler)
    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    
    # Save the best model based on F1-score
    best_model_idx = results_df['F1-Score'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'Model']
    print(f"\nBest Performing Model: {best_model_name}")
    
    best_model = best_estimators[best_model_name]
    
    # Package model and metadata
    model_data = {
        'model': best_model,
        'model_name': best_model_name,
        'scaler': scaler,
        'feature_names': feature_names.tolist(),
        'results': results_df,
        'all_models': best_estimators
    }
    
    joblib.dump(model_data, 'models/best_model.pkl')
    print("Updated best model and metadata saved to models/best_model.pkl")

if __name__ == "__main__":
    main()
