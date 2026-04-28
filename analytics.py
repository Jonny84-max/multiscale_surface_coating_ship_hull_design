# analytics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import ks_2samp

def load_and_train_model(csv_path):
    """Loads data, trains the model, and returns metrics + model object."""
    df = pd.read_csv(csv_path)
    
    features = ['riblet_height', 'riblet_spacing', 'lotus_intensity', 
                'velocity', 'temperature', 'salinity']
    target = 'drag_reduction'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Calculate Metrics
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mape": (mean_absolute_error(y_test, y_pred) / y_test.mean()) * 100,
        "ks_stat": ks_2samp(y_test, y_pred)[0],
        "p_value": ks_2samp(y_test, y_pred)[1]
    }
    
    return model, metrics, (y_test, y_pred), features

def create_reliability_plots(y_test, y_pred, p_value, features, feature_importances):
    """Generates the three-panel visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5, color='teal', s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_title('Actual vs Predicted')
    axes[0].set_xlabel('Measured Field Data (%)')
    axes[0].set_ylabel('Model Prediction (%)')

    # 2. KS Distribution (CDF)
    y_test_s, y_pred_s = np.sort(y_test), np.sort(y_pred)
    axes[1].plot(y_test_s, np.linspace(0, 1, len(y_test_s)), label='Field Data', color='blue')
    axes[1].plot(y_pred_s, np.linspace(0, 1, len(y_pred_s)), label='Model', color='red', linestyle='--')
    axes[1].set_title(f'KS Test (p-val: {p_value:.2f})')
    axes[1].legend()

    # 3. Feature Importance
    importances = pd.Series(feature_importances, index=features).sort_values(ascending=True)
    importances.plot(kind='barh', color='skyblue', ax=axes[2])
    axes[2].set_title('Multiscale Impact')
    
    plt.tight_layout()
    return fig
