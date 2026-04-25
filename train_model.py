import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the physics-refined dataset
data = pd.read_csv('biomimetic_dataset_physics_refined.csv')

# 2. Define Inputs (Features) and Outputs (Target)
# We focus on the engineered variables for better accuracy
X = data[['aspect_ratio', 'multiscale_index', 'effective_contact_angle', 
          'estimated_slip_length', 'velocity', 'time']]
y = data['drag_reduction']

# 3. Split data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
# Random Forest is highly effective for non-linear hydrodynamic data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Accuracy
predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)
error = mean_absolute_error(y_test, predictions)

print(f"--- Training Results ---")
print(f"Model Accuracy (R2): {accuracy:.4f}")
print(f"Average Error: {error:.2f}%")

# 6. Save the trained model to use in your predictive system
joblib.dump(model, 'shs_predictive_model.pkl')
print("\nModel saved as 'shs_predictive_model.pkl'")

# 7. Visualize Feature Importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importances.plot(kind='barh', title='What drives Drag Reduction?')
plt.tight_layout()
plt.show()