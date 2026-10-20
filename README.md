<h1 align="center">❤️ Heart Disease Prediction.</h1>

<p align="center">
  <i>Classifying whether a patient has heart disease from clinical measurements, comparing K-Nearest Neighbors (with hyperparameter tuning) against Gaussian Naive Bayes.</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

---

## 📋 Problem Statement

Heart disease is a leading cause of death, and early screening can save lives. Doctors look at many clinical indicators at once (age, blood pressure, cholesterol, ECG results and more), which is hard to do consistently by hand.

**Goal:** build a binary classifier that predicts whether a patient has heart disease (`target = 1`) from 13 clinical features, and find out which model and settings work best.

Because **missing a sick patient is costlier than a false alarm**, the models are tuned and compared with a focus on **recall**, alongside accuracy and precision.

## 📊 Dataset

| Property | Value |
|---|---|
| Samples | 303 patients |
| Features | 13 clinical features + 1 target |
| Missing values | None |
| Duplicate rows | 1 |
| Class balance | 165 with heart disease (`1`), 138 without (`0`), fairly balanced |

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Thalassemia result |
| `target` | **Label:** 1 = heart disease, 0 = no heart disease |

## 🔑 Pipeline

1. **Load data** with pandas and separate features (`X`) from the target (`y`)
2. **Train-test split:** 80% train (242), 20% test (61), `random_state=42`
3. **Scale features** with `StandardScaler`, fitted on the training set only (needed for distance-based KNN)
4. **Train KNN** for several values of *k* (3, 5, 7, 9)
5. **Tune with `GridSearchCV`:** 5-fold cross-validation, scored on **recall**
6. **Wrap in a `Pipeline`** (scaler + KNN) so scaling is done inside each CV fold and there is no data leakage
7. **Train Gaussian Naive Bayes** as a probabilistic baseline for comparison
8. **Evaluate** every model on the held-out test set using recall, accuracy and precision

## 📈 Results

### KNN: effect of *k*

| k | Recall | Accuracy | Precision |
|---|---|---|---|
| 3 | 0.781 | 0.852 | 0.926 |
| 5 | 0.875 | 0.902 | 0.933 |
| **7** | **0.906** | **0.918** | **0.935** |
| 9 | 0.875 | 0.902 | 0.933 |

### Cross-validated tuning (`GridSearchCV`, 5-fold, scoring = recall)

| k | Mean CV recall |
|---|---|
| 3 | 0.864 |
| 5 | 0.858 |
| **7** | **0.872** |
| 9 | 0.857 |

`GridSearchCV` independently picked **k = 7**, matching the best value found manually. The `Pipeline` version gives identical results.

### Model comparison (test set, 61 patients)

| Model | Recall | Accuracy | Precision |
|---|---|---|---|
| Gaussian Naive Bayes | 0.844 | 0.869 | 0.900 |
| **KNN (k = 7, tuned)** | **0.906** | **0.918** | **0.935** |

🏆 **Best model: KNN with k = 7**, with 90.6% recall, 91.8% accuracy and 93.5% precision. It catches about 9 in 10 patients with heart disease in the test set.

**Takeaways**
- Too small a *k* (3) overfits to noise, and performance is best at a moderate value (7), then dips again at 9.
- KNN beats Naive Bayes on every metric here. Naive Bayes assumes the features are independent, which is not true for clinical data (for example age, blood pressure and cholesterol are related).
- Naive Bayes is trained on unscaled data on purpose, since `GaussianNB` does not need scaling.

## 🛠️ Installation

**Prerequisites:** Python 3.8+

```bash
# Clone the repository
git clone https://github.com/ShivamChavan2311/heart-disease-prediction.git
cd heart-disease-prediction

# Install dependencies
pip install pandas scikit-learn jupyter
```

## 🚀 Usage

Run the notebooks:

```bash
jupyter notebook knn.ipynb          # KNN, tuning and pipeline
jupyter notebook naive_bayes.ipynb  # Naive Bayes baseline
```

Or use the final tuned pipeline directly:

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score

heart_df = pd.read_csv("heart.csv")
X = heart_df.drop("target", axis=1)
y = heart_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier()),
])

search = GridSearchCV(
    pipeline,
    param_grid={"knn__n_neighbors": [3, 5, 7, 9]},
    cv=5,
    scoring="recall",
)
search.fit(X_train, y_train)

y_pred = search.predict(X_test)
print("Best params:", search.best_params_)
print("Recall:   ", recall_score(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
```

## 📁 Project Structure

```
heart-disease-prediction/
│
├── heart.csv            # Dataset (303 patients, 14 columns)
├── knn.ipynb            # KNN, k comparison, GridSearchCV, Pipeline
├── naive_bayes.ipynb    # Gaussian Naive Bayes baseline
└── README.md            # This file
```

## 📚 What I Learned

- Why distance-based models like KNN need scaled features
- How *k* affects the bias-variance trade-off in KNN
- Hyperparameter tuning with `GridSearchCV` and cross-validation
- Using a `Pipeline` so preprocessing is fitted inside each fold and avoids data leakage
- Choosing a metric that fits the problem (recall for screening) instead of defaulting to accuracy
- How a simple baseline (Naive Bayes) helps judge a more flexible model

## ⚠️ Limitations

- The test set has only **61 patients**, so one patient changes accuracy by about 1.6 points. The gap between KNN and Naive Bayes is only a few patients and could shift with a different split.
- The first comparison of *k* values was judged on the test set. The `GridSearchCV` step fixes this by choosing *k* with cross-validation on training data only, and it agrees on k = 7.
- Naive Bayes was evaluated on a single split and was not cross-validated, so the comparison is indicative rather than definitive.
- The dataset has 1 duplicate row, and the split is not stratified.
- **This is a learning project and is not a medical tool.** It must not be used for real diagnosis.

## 🔭 Future Improvements

- Use stratified splits and cross-validate **every** model for a fair comparison
- Add a confusion matrix and ROC-AUC curve
- Try Logistic Regression, Random Forest and Gradient Boosting
- Tune more KNN settings (`weights`, `metric`) and a wider range of *k*
- Add feature-importance analysis and a small Streamlit app for live predictions

## 👤 Author

**Shivam Chavan**

[LinkedIn](https://www.linkedin.com/in/shivamchavansppu/) · [Portfolio](https://shivamchavan2311.github.io/portfolio/) · [GitHub](https://github.com/ShivamChavan2311) · [chavanshivam160@gmail.com](mailto:chavanshivam160@gmail.com)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center"><b>⭐ If you found this useful, consider starring the repo.</b></p>
