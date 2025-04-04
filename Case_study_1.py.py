import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Changed from RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

data = pd.read_csv("solar.csv")
print("Data Preview")
print(data.head())
print("Data Info")
print(data.info())

# feature selection
features = ["temperature", "voltage", "current", "days_since_maintenance"]
target = "failure"

missing_cols = [col for col in features + [target] if col not in data.columns]
if missing_cols:
    raise ValueError(f"Missing columns in csv : {missing_cols}")

x = data[features]
y = data[target]

if x.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    print("Dropping rows with missing values to increase accuracy ! ")
    data = data.dropna(subset=features + [target])
    x = data[features]
    y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Changed to SVM with probability=True to enable predict_proba
model = SVC(kernel='rbf', probability=True, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy = {accuracy:.2f}")

print(f"Classification report : ")
print(classification_report(y_test, y_pred))

# For SVM, we can use feature importance from permutation importance as direct feature_importances_ isn't available
from sklearn.inspection import permutation_importance
perm_importance = permutation_importance(model, x_test, y_test, n_repeats=10, random_state=42)
feature_importance = pd.Series(perm_importance.importances_mean, index=features)
print("\n Feature importances : ")
print(feature_importance)

feature_importance.plot(kind="bar", title="Feature Importance for Predicting inverter failure (SVM)")
plt.ylabel("Importance")
plt.show()

sample = x_test.iloc[0].values.reshape(1, -1)
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print(f"\n Sample Prediction : ")
print(f"Features : {x_test.iloc[0].to_dict()}")
print(f"Predicted failure : {'yes' if prediction == 1 else 'no'} (Probability : {probability:.2f})")

if prediction == 1:
    print("Action : schedule immediate maintenance for inverter")
    print("    - Check temperature regulating system (Possible overheating !)")
    print("    - Inspect voltage and current stability (Possible electric fault !)")
    print("    - Review maintenance logs and reset schedules (long gap still last maintenance !)")
else:
    print("Action : no immediate action required. Continue Monitoring.")
