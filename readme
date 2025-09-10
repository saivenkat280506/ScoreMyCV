## ScoreMyCV ✨

Transform raw resumes into meaningful scores with a clean, reproducible ML pipeline.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data%20Frames-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/status-experimental-yellow)

---

### Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Data Format](#data-format)
- [Generate Train/Test Split](#generate-traintest-split-optional)
- [Train & Sample Prediction](#train-the-model-and-run-a-sample-prediction)
- [Use the Saved Model](#use-the-saved-model-for-your-own-predictions)
- [Notes & Tips](#notes-and-tips)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

### Overview
ScoreMyCV is a lightweight machine learning project that predicts a numeric score for a resume using:
- **Category**: the role (e.g., "Backend Developer")
- **Resume**: free‑text resume content

Under the hood, it uses a scikit‑learn `Pipeline` with:
- One‑Hot Encoding for `Category`
- TF‑IDF Vectorization for `Resume`
- RandomForestRegressor to predict the score

The trained pipeline is saved to `score_model.pkl` for easy reuse.

> 💡 Ideal for learning, experimentation, and quick prototyping.

---

### Features
- 🔤 TF‑IDF text vectorization (unigrams, 3000 max features, English stop‑words)
- 🧩 One‑Hot encoding for categorical role labels
- 🌲 Robust baseline with Random Forest regression
- 💾 Single‑file model artifact via `joblib`
- ⚡ Simple scripts and JSON data for quick iteration

---

### Project Structure
- `ScoreMyCV.py`: trains the model and makes a sample prediction
- `data_gen.py`: splits a combined dataset into `train_data.json`/`test_data.json`
- `train_data.json`: training samples (JSON list)
- `test_data.json`: held‑out samples (JSON list)
- `score_model.pkl`: saved scikit‑learn pipeline (created after training)
- `app/build.gradle`: placeholder (not used by the Python pipeline)
- `readme`: this file

---

### Requirements
- Python 3.9+
- pip

Python packages:
- scikit‑learn
- pandas
- joblib

Install dependencies:
```bash
pip install scikit-learn pandas joblib
```

Use a virtual environment (recommended):
```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install scikit-learn pandas joblib
```

---

### Data Format
`train_data.json` and `test_data.json` are JSON arrays of objects.

- Required for training: `Category` (string), `Resume` (string), `Score` (number)
- Example item:
```json
{
  "Category": "Backend Developer",
  "Resume": "Experienced in REST APIs, Python, Django, PostgreSQL",
  "Score": 78.5
}
```

> ✅ At inference time, only `Category` and `Resume` are needed.

---

### Generate Train/Test Split (optional)
If you have a combined dataset in `train_data.json`, create a split via:
```bash
python data_gen.py
```
This rewrites `train_data.json` with ~80% of the data and creates `test_data.json` with ~20%.

---

### Train the Model and Run a Sample Prediction
Run:
```bash
python ScoreMyCV.py
```
This will:
- Load `train_data.json`
- Build and fit the pipeline
- Save the model to `score_model.pkl`
- Load the saved model and print a predicted score for a hardcoded example resume

---

### Use the Saved Model for Your Own Predictions
```python
import joblib
import pandas as pd

model = joblib.load("score_model.pkl")

samples = [
    {"Category": "Backend Developer", "Resume": "Python, FastAPI, PostgreSQL, Docker, AWS"},
    {"Category": "Data Scientist", "Resume": "Pandas, Scikit-Learn, NLP, XGBoost, ML Ops"}
]

df = pd.DataFrame(samples)
preds = model.predict(df)
print(preds)
```

---

### Notes and Tips
- The TF‑IDF vectorizer uses unigrams with up to 3000 features and English stop‑words.
- RandomForest parameters can be tuned in `ScoreMyCV.py` (e.g., `n_estimators`, `random_state`).
- `handle_unknown='ignore'` is set for the encoder, so unseen categories at inference are safely ignored.

---

### Troubleshooting
- On Windows, ensure you have a recent `pip` and a compatible Python version. Wheels are usually available for scikit‑learn.
- Make sure `train_data.json` is valid JSON with the required fields.
- If you see encoding issues, ensure files are UTF‑8 encoded.

---

### License
No license specified. Add one if you plan to share or open‑source this project.

### Acknowledgments
Built with ❤️ using scikit‑learn, pandas, and joblib.
