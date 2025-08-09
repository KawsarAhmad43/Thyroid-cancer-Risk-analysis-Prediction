import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Models
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from xgboost import XGBClassifier

def run_pipeline_with_cm():
    # 1. Load Data
    try:
        df = pd.read_csv("minified_dataset.csv", encoding='utf-8-sig')
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

    # 3. Define Models
    rf_clf = RandomForestClassifier(random_state=42)
    gbm_clf = GradientBoostingClassifier(random_state=42)
    et_clf = ExtraTreesClassifier(random_state=42)
    svm_clf = SVC(probability=True, random_state=42)
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    vc_clf = VotingClassifier(
        estimators=[('rf', rf_clf), ('gbm', gbm_clf), ('et', et_clf), ('svm', svm_clf), ('xgb', xgb_clf)],
        voting='soft'
    )

    models = {
        "Random Forest": rf_clf,
        "Gradient Boosting": gbm_clf,
        "Extra Trees": et_clf,
        "SVM": svm_clf,
        "XGBoost": xgb_clf,
        "Voting Classifier": vc_clf,
    }

    results = {}
    roc_plt = plt.figure(figsize=(10, 8))

    all_cms = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        # 4. Train and Evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Store results
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"--- {name} ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # Store confusion matrix
        all_cms[name] = confusion_matrix(y_test, y_pred)
        print("-" * 20)

        # ROC Curve data
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    # Finalize ROC plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curves.png")
    print("ROC curve plot saved as roc_curves.png")
    plt.close(roc_plt)

    # Plot Confusion Matrices
    cm_rows = int(np.ceil(len(all_cms) / 3))
    cm_fig, axes = plt.subplots(cm_rows, 3, figsize=(15, cm_rows * 5))
    axes = axes.flatten()

    for i, (name, cm) in enumerate(all_cms.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'CM for {name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        cm_fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    print("Confusion matrices plot saved as confusion_matrices.png")
    plt.close(cm_fig)


if __name__ == "__main__":
    run_pipeline_with_cm()
