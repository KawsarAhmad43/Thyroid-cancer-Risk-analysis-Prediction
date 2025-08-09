import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import warnings
from lightgbm import LGBMClassifier

# Suppress warnings
warnings.filterwarnings('ignore')

def run_lgbm_pipeline():
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

    # 3. Define Model and Hyperparameter Grid
    model = LGBMClassifier(random_state=42)
    params = {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__num_leaves': [20, 30],
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    # Use RandomizedSearchCV for faster tuning
    random_search = RandomizedSearchCV(
        pipeline, params, n_iter=4, cv=2, n_jobs=-1, scoring='accuracy', random_state=42
    )

    print("--- Tuning LightGBM ---")
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Best parameters for LightGBM: {random_search.best_params_}")
    print(f"Accuracy for LightGBM: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'LightGBM (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Tuned LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curve_lgbm.png")
    print("Tuned ROC curve plot saved as roc_curve_lgbm.png")

if __name__ == "__main__":
    run_lgbm_pipeline()
