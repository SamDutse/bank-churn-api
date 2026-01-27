# train.py
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/Churn_Modelling.csv")

# Drop irrelevant columns
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

X = df.drop("Exited", axis=1)
y = df["Exited"]

# Column groups
numeric_features = [
    "CreditScore", "Age", "Tenure",
    "Balance", "NumOfProducts",
    "EstimatedSalary"
]

categorical_features = ["Geography", "Gender"]

# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save pipeline
joblib.dump(pipeline, "model/churn_pipeline.pkl")

print("Model pipeline saved successfully")
