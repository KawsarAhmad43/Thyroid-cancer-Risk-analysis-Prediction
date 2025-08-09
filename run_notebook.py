import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# --- Data Loading ---
# Modified to use the local minified dataset
df = pd.read_csv("minified_dataset.csv", encoding='utf-8-sig')

# --- Preprocessing ---
if "Patient_ID" in df.columns:
    df = df.drop("Patient_ID", axis=1)

# The notebook has some strange and conflicting preprocessing.
# I will use a simplified and clean version of what's there.
# This part handles missing values and encodes all object columns.
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Separate X and y
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Scaling ---
# The notebook scales the whole dataframe, which is not ideal.
# It's better to fit the scaler on the training data and transform both train and test.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# --- Model Training ---
# Adding SVC as requested
models_params = {
    'RandomForest': (RandomForestClassifier(random_state=42), {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
    }),
    'LogisticRegression': (LogisticRegression(random_state=42, max_iter=500), {
        'C': [0.01, 1, 100],
    }),
    'XGBoost': (XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
    }),
    'SVM': (SVC(random_state=42, probability=True), {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    })
}

results = {}
all_best_models = {}
plt.figure(figsize=(10, 8))

print("Starting model training...")
for name, (model, param_dist) in models_params.items():
    print(f"Training {name}...")

    # Using a small n_iter to avoid timeouts
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=3,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    all_best_models[name] = best_model
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"--- {name} Results ---")
    print(f"Best Params: {search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, y_pred))

    # ROC Curve
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# --- Final ROC Plot ---
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Models')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig("notebook_roc_curve.png")
print("\nROC curve plot saved as notebook_roc_curve.png")

# --- Final Accuracy Summary ---
print("\n--- Model Accuracy Summary ---")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
