import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import ks_2samp

def run_reliability_study(csv_path):
    """
    Performs the full statistical validation of the underlying model 
    using the provided dataset.
    """
    df = pd.read_csv(csv_path)
    
    # Feature set used for training
    features = ['riblet_height', 'riblet_spacing', 'lotus_intensity', 
                'velocity', 'temperature', 'salinity']
    target = 'drag_reduction'
    
    X = df[features]
    y = df[target]

    # Split into Train and Field (Test) data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train internal validation model
    val_model = RandomForestRegressor(n_estimators=100, random_state=42)
    val_model.fit(X_train, y_train)
    y_pred = val_model.predict(X_test)

    # Calculate metrics
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mape": (mean_absolute_error(y_test, y_pred) / y_test.mean()) * 100,
        "ks_stat": ks_2samp(y_test, y_pred)[0],
        "p_value": ks_2samp(y_test, y_pred)[1]
    }
    
    # Prepare the 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Graph 1: Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.5, color='teal', s=10)
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_title(f'Actual vs Predicted\n($R^2 = {metrics["r2"]:.3f}$)')
    axes[0].set_xlabel('Measured Field Data (%)')
    axes[0].set_ylabel('Model Prediction (%)')
    axes[0].grid(True, alpha=0.3)

    # Graph 2: KS Distribution (CDF)
    y_test_sorted = np.sort(y_test)
    y_pred_sorted = np.sort(y_pred)
    axes[1].plot(y_test_sorted, np.linspace(0, 1, len(y_test_sorted)), label='Field Data', color='blue', lw=2)
    axes[1].plot(y_pred_sorted, np.linspace(0, 1, len(y_pred_sorted)), label='Model', color='red', linestyle='--', lw=2)
    axes[1].set_title(f'KS Test CDF\n(p-value = {metrics["p_value"]:.2f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Graph 3: Feature Importance
    importances = pd.Series(val_model.feature_importances_, index=features).sort_values(ascending=True)
    importances.plot(kind='barh', color='skyblue', ax=axes[2])
    axes[2].set_title('Multiscale Parameter Impact')
    
    plt.tight_layout()
    
    return metrics, fig
