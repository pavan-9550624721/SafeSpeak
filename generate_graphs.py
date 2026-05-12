import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
)

from feature_engineering import run_feature_engineering

# 1. Setup folders
os.makedirs("Model_Graphs/XGBoost", exist_ok=True)
os.makedirs("Model_Graphs/Random_Forest", exist_ok=True)

# 2. Get Data
print("[INFO] Loading data and engineering features...")
X_train_sparse, X_val_sparse, y_train, y_val, feature_names = run_feature_engineering(
    "train.csv"
)
target_cols = y_train.columns.tolist()


def plot_confusion_matrices(y_v, y_p, labels, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, label in enumerate(labels):
        cm = confusion_matrix(y_v.iloc[:, i], y_p[:, i])
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[i], cmap="Blues")
        axes[i].set_title(f"Confusion Matrix - {label}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc(y_v, y_prob, labels, path):
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_v.iloc[:, i], y_prob[:, i])
        auc_score = roc_auc_score(y_v.iloc[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_pr(y_v, y_prob, labels, path):
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(y_v.iloc[:, i], y_prob[:, i])
        plt.plot(recall, precision, label=label)
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metrics_bar(y_v, y_p, labels, path, model_name):
    # Overall metrics
    acc = accuracy_score(y_v, y_p)
    micro_f1 = f1_score(y_v, y_p, average="micro")
    macro_f1 = f1_score(y_v, y_p, average="macro")

    plt.figure(figsize=(8, 6))
    metrics = ["Exact Match Accuracy", "Micro F1", "Macro F1"]
    scores = [acc, micro_f1, macro_f1]
    sns.barplot(x=metrics, y=scores, palette="viridis")
    plt.ylim(0, 1.0)
    for i, v in enumerate(scores):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")
    plt.title(f"{model_name} Overall Performance Metrics")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    # Per label precision/recall
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.35

    precisions = [
        precision_score(y_v.iloc[:, i], y_p[:, i]) for i in range(len(labels))
    ]
    recalls = [recall_score(y_v.iloc[:, i], y_p[:, i]) for i in range(len(labels))]

    plt.bar(x - width / 2, precisions, width, label="Precision", color="skyblue")
    plt.bar(x + width / 2, recalls, width, label="Recall", color="salmon")
    plt.xticks(x, labels)
    plt.ylim(0, 1.1)
    for i, v in enumerate(precisions):
        plt.text(x[i] - width / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(recalls):
        plt.text(x[i] + width / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

    plt.title(f"{model_name} Per-Label Precision & Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path.replace(".png", "_per_label.png"))
    plt.close()


# --- XGBOOST ---
print("\n" + "=" * 50)
print("[INFO] Training XGBoost...")
xgb = OneVsRestClassifier(
    XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        random_state=42,
    )
)
xgb.fit(X_train_sparse, y_train)

print("[INFO] Generating XGBoost Graphs...")
y_prob_xgb = xgb.predict_proba(X_val_sparse)
y_pred_xgb = (y_prob_xgb >= 0.5).astype(int)

plot_confusion_matrices(
    y_val, y_pred_xgb, target_cols, "Model_Graphs/XGBoost/confusion_matrices.png"
)
plot_roc(y_val, y_prob_xgb, target_cols, "Model_Graphs/XGBoost/roc_curves.png")
plot_pr(y_val, y_prob_xgb, target_cols, "Model_Graphs/XGBoost/pr_curves.png")
plot_metrics_bar(
    y_val,
    y_pred_xgb,
    target_cols,
    "Model_Graphs/XGBoost/accuracy_precision_bars.png",
    "XGBoost",
)

# --- RANDOM FOREST ---
print("\n" + "=" * 50)
print("[INFO] Training Random Forest...")
rf = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
)
rf.fit(X_train_sparse, y_train)

print("[INFO] Generating Random Forest Graphs...")
y_prob_rf = rf.predict_proba(X_val_sparse)
y_pred_rf = (y_prob_rf >= 0.5).astype(int)

plot_confusion_matrices(
    y_val, y_pred_rf, target_cols, "Model_Graphs/Random_Forest/confusion_matrices.png"
)
plot_roc(y_val, y_prob_rf, target_cols, "Model_Graphs/Random_Forest/roc_curves.png")
plot_pr(y_val, y_prob_rf, target_cols, "Model_Graphs/Random_Forest/pr_curves.png")
plot_metrics_bar(
    y_val,
    y_pred_rf,
    target_cols,
    "Model_Graphs/Random_Forest/accuracy_precision_bars.png",
    "Random Forest",
)

print("\n[SUCCESS] All graphs generated and saved to 'Model_Graphs' folder.")
