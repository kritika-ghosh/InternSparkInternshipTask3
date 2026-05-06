# 🚢 Titanic Survival Prediction Pipeline

An end-to-end, production-ready machine learning pipeline for predicting passenger survival on the Titanic. Built using robust data preprocessing, domain-specific feature engineering, and explainable AI techniques, this pipeline benchmarks multiple classification algorithms to deliver a highly accurate and interpretable final predictor.

---

## 📌 Project Overview
Predicting passenger survival on the Titanic is a classic binary classification challenge. Historical records show that evacuation prioritized specific demographics ("women and children first"), structural placement on the ship (upper decks vs. lower decks), and social status.

This project delivers:
1. **Exploratory Data Analysis (EDA):** In-depth univariate, bivariate, and multivariate analysis mapping out demographic and socioeconomic survival splits.
2. **Feature Engineering:** Advanced text-parsing and numerical transformations extracting structural signals (`Title`, `FamilySize`, and `HasCabin`).
3. **Pre-processing Pipelines:** Fully integrated `ColumnTransformer` automating median imputation for missing numerical scales, mode imputation, standardization, and categorical encoding.
4. **Algorithm Benchmarking:** Rigorous comparison between Logistic Regression, $k$-Nearest Neighbors ($k$-NN), Decision Trees, and Random Forests.
5. **Model Explainability & Serialization:** Feature importance reporting alongside fully serialized preprocessing and estimator binaries for seamless integration.

---

## 📊 Dataset Schema & Features
The pipeline processes the standard Titanic passenger manifest containing the following raw attributes:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| **`Survived`** | `int64` | **Target Variable** (0 = Deceased, 1 = Survived) |
| **`Pclass`** | `int64` | Ticket Class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| **`Name`** | `object` | Full passenger name (including structural prefixes) |
| **`Sex`** | `object` | Passenger gender (male/female) |
| **`Age`** | `float64` | Passenger age (contains missing values) |
| **`SibSp`** | `int64` | Number of siblings or spouses aboard the Titanic |
| **`Parch`** | `int64` | Number of parents or children aboard the Titanic |
| **`Ticket`** | `object` | Ticket serial number |
| **`Fare`** | `float64` | Passenger fare price |
| **`Cabin`** | `object` | Cabin number (highly sparse; contains missing values) |
| **`Embarked`** | `object` | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 🛠️ Pipeline Architecture

### 1. Feature Engineering
Raw passenger variables are processed to construct highly descriptive structural signals:
* **`Title` Extraction:** Standardizes rare and variable prefixes from the `Name` attribute into clean social groups:
  * *Rules:* `Mlle` $\\rightarrow$ `Miss`, `Mme` $\\rightarrow$ `Mrs`, `Capt`/`Col`/`Major` $\\rightarrow$ `Officer`, `Countess`/`Don`/`Sir` $\\rightarrow$ `Royal`.
* **`FamilySize`:** Unifies immediate and extended family counts to identify single travelers versus large, slow-evacuating family networks:
  $$\\text{FamilySize} = \\text{SibSp} + \\text{Parch} + 1$$
* **`HasCabin`:** Converts the highly sparse `Cabin` text column into a binary indicator (1 if a cabin was explicitly recorded, 0 otherwise), acting as a proxy for upper-deck placement:
  $$\\text{HasCabin} = \\begin{cases} 1 & \\text{if Cabin is not null} \\\\ 0 & \\text{otherwise} \\end{cases}$$

### 2. Preprocessing & Column Transformations
The pipeline constructs automated pathways using a `ColumnTransformer` to enforce data hygiene and prevent feature leakage during splitting:
* **Numerical Features (`Age`, `Fare`, `FamilySize`):**
  * Imputed using `SimpleImputer(strategy='median')`.
  * Normalized to unit variance using `StandardScaler()`.
* **Categorical Features (`Sex`, `Embarked`, `Title`, `Pclass`, `HasCabin`):**
  * Imputed using `SimpleImputer(strategy='most_frequent')`.
  * Encoded using `OneHotEncoder(drop='first', handle_unknown='ignore')` to eliminate multicollinearity.

---

## 📈 Model Performance & Comparison
Four classifiers are trained and benchmarked using $80/20$ stratified splits:

| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| **Logistic Regression** | ~80.5% | ~0.74 |
| **K-Nearest Neighbors (k-NN)** | ~79.0% | ~0.71 |
| **Decision Tree** | ~81.0% | ~0.75 |
| **Random Forest (Best)** | **~83.5%** | **~0.78** |

*Note: Performance statistics vary slightly based on training dataset allocations. The Random Forest ensemble consistently minimizes variance and generalizes best on test distributions.*

---

## 🚀 Execution & Inference

### ⚙️ Prerequisites
Ensure you have the required packages installed in your environment:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### 🏃 Running the Pipeline
Execute the main script. The script is designed to be fully self-sustaining. If `titanic.csv` is not present, it automatically creates mock passenger data mimicking the target and features, allowing immediate executable testing:
```bash
python titanic_pipeline.py
```

### 💾 Generated Artifacts
Upon execution, the script generates and saves the following serialized pipeline configurations:
* `titanic_preprocessor.pkl`: The exact preprocessor configuration (imputers, scalers, and encoders).
* `titanic_rf_model.pkl`: The optimized Random Forest classifier weights.

---

## 🔮 Production Inference Example
Use the following Python snippet to load the serialized artifacts and perform predictions on new passenger records:

```python
import joblib
import pandas as pd
import numpy as np

# 1. Load the serialized pipeline artifacts
model = joblib.load('titanic_rf_model.pkl')
preprocessor = joblib.load('titanic_preprocessor.pkl')

def predict_passenger_survival(pclass, sex, age, sibsp, parch, fare, embarked, cabin):
    # Rule heuristic to extract Title
    title = 'Mr'
    if sex == 'female':
        title = 'Mrs'

    # Map input attributes into a raw DataFrame matching the training format
    new_input = pd.DataFrame([{
        'Age': age,
        'Fare': fare,
        'SibSp': sibsp,
        'Parch': parch,
        'Pclass': pclass,
        'Sex': sex,
        'Embarked': embarked,
        'Cabin': cabin,
        'Title': title
    }])

    # Replicate engineered variables
    new_input['FamilySize'] = new_input['SibSp'] + new_input['Parch'] + 1
    new_input['HasCabin'] = new_input['Cabin'].notna().astype(int)

    # Reorder columns to match the preprocessor specifications
    features_to_transform = ['Age', 'Fare', 'FamilySize', 'Sex', 'Embarked', 'Title', 'Pclass', 'HasCabin']
    
    # Scale and encode raw variables
    scaled_input = preprocessor.transform(new_input[features_to_transform])

    # Infer predictions
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    status = "Survived" if prediction == 1 else "Did Not Survive"
    return status, probability

# --- Test Case 1: High-Survival Profile (35yo 1st Class Female) ---
status_1, prob_1 = predict_passenger_survival(
    pclass=1, sex='female', age=35.0, sibsp=1, parch=0, fare=150.0, embarked='C', cabin='C85'
)
print(f"Passenger 1: {status_1} (Survival Probability: {prob_1 * 100:.2f}%)")

# --- Test Case 2: Low-Survival Profile (22yo 3rd Class Male) ---
status_2, prob_2 = predict_passenger_survival(
    pclass=3, sex='male', age=22.0, sibsp=1, parch=0, fare=7.25, embarked='S', cabin=None
)
print(f"Passenger 2: {status_2} (Survival Probability: {prob_2 * 100:.2f}%)")
```
