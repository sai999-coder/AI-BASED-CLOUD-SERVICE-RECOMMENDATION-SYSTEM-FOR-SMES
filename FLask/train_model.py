import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv("cloud_full_dataset.csv")

# Encode categorical columns
le_dict = {}
cat_cols = ["budget", "storage", "scalability", "service", "recommended_provider"]

for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

# Features
X = data[["budget", "storage", "scalability", "service"]]

# Outputs
y_reg = data[["price", "speed", "storage_space"]]
y_clf = data["recommended_provider"]

# Models
reg_model = RandomForestRegressor()
clf_model = DecisionTreeClassifier()

reg_model.fit(X, y_reg)
clf_model.fit(X, y_clf)

# Save models
with open("model.pkl", "wb") as f:
    pickle.dump((reg_model, clf_model, le_dict), f)

print("Model trained successfully!")
