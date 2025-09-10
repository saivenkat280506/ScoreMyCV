import json
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


print("Loading training data...")
with open("train_data.json", "r") as f:
    train_data = json.load(f)
train_df = pd.DataFrame(train_data)
print(f"Loaded {len(train_df)} training resumes")
X_train = train_df[["Category", "Resume"]]
y_train = train_df["Score"]
print("Building pipeline...")
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown='ignore'), ["Category"]),
    ("text", TfidfVectorizer(ngram_range=(1, 1), max_features=3000, stop_words='english'), "Resume")
])
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42))
])
print("Training model...")
model.fit(X_train, y_train)
print("Model training complete")
joblib.dump(model, "score_model.pkl")
print("Model saved as score_model.pkl")
print("Predicting on custom resume...")

test_resume = {
"Category": "Backend Developer",
"Resume": "Beth Hayes skilled in rest api, pandas, redux, express, mongodb, rest api, mysql, java. Holds a degree from Johnson LLC University."

}
test_df = pd.DataFrame([test_resume])
loaded_model = joblib.load("score_model.pkl")
predicted_score = loaded_model.predict(test_df)[0]
print(f"Predicted Score for test resume: {predicted_score:.2f}")    





# To test the model here are few samples


