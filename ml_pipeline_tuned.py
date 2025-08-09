import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import warnings

# Models
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Suppress warnings
warnings.filterwarnings('ignore')

def run_tuned_pipeline():
    # 1. Load Data
    try:
        df = pd.read_csv("minified_dataset.csv")
    except FileNotFoundError:
        print("Error: minified_dataset.csv not found. Please make sure the file is in the correct directory.")
        return

    # 2. Preprocess Data
    if "Patient_ID" in df.columns:
        df = df.drop("Patient_ID", axis=1)

    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column])

    X = df.drop("Diagnosis", axis=1)
    y = df["Diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Define Models and Hyperparameter Grids
    models_and_params = {
        "Extra Trees": (ExtraTreesClassifier(random_state=42), {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
        }),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
        }),
        "SVM": (SVC(probability=True, random_state=42), {
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': ['scale', 'auto'],
        }),
        "CatBoost": (CatBoostClassifier(random_state=42, verbose=0), {
            'classifier__iterations': [100, 200],
            'classifier__depth': [4, 6],
        }),
        "LightGBM": (LGBMClassifier(random_state=42), {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
        }),
    }

    results = {}
    plt.figure(figsize=(12, 10))

    for name, (model, params) in models_and_params.items():
        print(f"--- Tuning {name} ---")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(pipeline, params, cv=3, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Accuracy for {name}: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        print("-" * 20)

        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    # Finalize ROC plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Tuned Models ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curves_tuned.png")
    print("Tuned ROC curve plot saved as roc_curves_tuned.png")

if __name__ == "__main__":
    run_tuned_pipeline()
