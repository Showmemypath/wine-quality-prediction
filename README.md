# Wine Quality Prediction

Binary classification model predicting whether a red wine is "good quality" (quality ≥ 7) using physicochemical features.

## Dataset
[Red Wine Quality Dataset](https://www.kaggle.com/datasets/sumit17125/red-wine-quality-dataset) — 1,599 samples, 11 physicochemical features (acidity, sugar, sulphates, alcohol, etc.) plus a quality score (3–8).

## Approach

**EDA**
- Univariate analysis on the target and all features — found heavy class imbalance in `quality` and skew/outliers across most features
- Split the data into top 10% vs bottom 90% by quality to visually compare distributions and spot features that separate good wine from the rest
- Bivariate analysis via correlation heatmap — `alcohol` and `volatile acidity` showed the highest correlation with quality

**Preprocessing**
- Reframed the problem as binary classification (`quality >= 7` → 1, else 0) to address the severe class imbalance across the original 6-class target
- Selected `alcohol`, `citric acid`, and `sulphates` as features based on EDA — the features with the most noticeable distributional difference between high and low quality wines

**Modeling**
Trained and tuned four classifiers via `GridSearchCV`, comparing on precision, recall, and F1-score (given the class imbalance, accuracy alone would be misleading):
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Random Forest
- Logistic Regression (with `class_weight='balanced'`)

## Results

| Model | Accuracy | Good Wine Precision | Good Wine Recall |
|---|---|---|---|
| KNN | 0.88 | 0.66 | 0.45 |
| Naive Bayes | 0.85 | 0.50 | 0.30 |
| Random Forest | 0.88 | 0.61 | 0.47 |
| Logistic Regression | 0.79 | 0.40 | 0.83 |

**Key observations**
- Class imbalance means all models predict the majority class (low/mid quality) well; the real differentiation is in detecting good wine
- KNN and Random Forest perform comparably on the minority class — KNN gives fewer false positives, Random Forest gives fewer false negatives
- Logistic Regression trades precision for recall, catching far more good wines at the cost of more false positives

## Tools
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Future Work
- Try additional models — SVM, Gradient Boosting (XGBoost/LightGBM/CatBoost)
- Address class imbalance more directly (SMOTE, threshold tuning)
- Hyperparameter tuning beyond current grid search ranges

## How to Run

```bash
git clone https://github.com/Showmemypath/wine-quality-prediction.git
cd wine-quality-prediction
pip install -r requirements.txt
```

Open `wine-quality.ipynb` in Jupyter and run all cells. The dataset downloads automatically via `kagglehub`.

## Project Structure

```
├── wine-quality.ipynb   # EDA, preprocessing, model training and evaluation
├── requirements.txt
└── README.md
```