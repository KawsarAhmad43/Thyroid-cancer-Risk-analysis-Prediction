import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import warnings
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import time

# Suppress warnings
warnings.filterwarnings('ignore')

def run_final_pipeline():
    print("Script started.")
    start_time = time.time()

    # 1. Load Data
    print("Loading data...")
    try:
        df = pd.read_csv("minified_dataset.csv", encoding='utf-8-sig')
    except FileNotFoundError:
        print("Error: minified_dataset.csv not found.")
        return
    print(f"Data loaded in {time.time() - start_time:.2f} seconds.")

    # 2. Preprocess Data
    print("Preprocessing data...")
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
    print(f"Data preprocessed in {time.time() - start_time:.2f} seconds.")

    # 3. Manually Set Hyperparameters and Define Models
    lgbm = LGBMClassifier(
        random_state=42,
        n_estimators=100,  # Reduced n_estimators
        learning_rate=0.1,
        num_leaves=20
    )

    xgb = XGBClassifier(
        random_state=42,
        n_estimators=100,  # Reduced n_estimators
        learning_rate=0.1,
        eval_metric='logloss',
        use_label_encoder=False
    )

    # 4. Create a Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[('lgbm', lgbm), ('xgb', xgb)],
        voting='soft'
    )

    models = {
        "LightGBM": lgbm,
        "XGBoost": xgb,
        "Voting Classifier": voting_clf
    }

    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        print(f"--- Training {name} ---")
        train_start_time = time.time()
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)
        print(f"Trained {name} in {time.time() - train_start_time:.2f} seconds.")

        eval_start_time = time.time()
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {name}: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))
        print("-" * 20)
        print(f"Evaluated {name} in {time.time() - eval_start_time:.2f} seconds.")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    # Finalize ROC plot
    print("Saving ROC plot...")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Final Models ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curves_final.png")
    print("Final ROC curve plot saved as roc_curves_final.png")
    print(f"Script finished in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    run_final_pipeline()
