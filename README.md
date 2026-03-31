# Fundamentals-of-AI-and-ML
"""
Crop Disease Detector
=====================
Fundamentals of AI and ML - BYOP Capstone Project

Detects crop diseases from feature-based data using Random Forest Classifier.
Supports: Tomato, Corn, Rice, Wheat, Potato
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print("=" * 55)
    print("  CROP DISEASE DETECTOR — Data Overview")
    print("=" * 55)
    print(f"  Total samples   : {len(df)}")
    print(f"  Features        : {df.shape[1] - 1}")
    print(f"  Crops covered   : {df['crop_type'].nunique()}")
    print(f"  Disease classes : {df['disease_label'].nunique()}")
    print("=" * 55)
    print("\nClass distribution:")
    print(df['disease_label'].value_counts().to_string())
    print()
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESS
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """Encode categoricals and split into X / y."""
    df = df.copy()

    # Drop non-feature column
    df.drop(columns=["image_id"], inplace=True)

    # Columns to label-encode
    cat_cols = ["crop_type", "leaf_color", "spot_size",
                "spot_color", "yellowing", "wilting"]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Target encoder
    target_le = LabelEncoder()
    y = target_le.fit_transform(df["disease_label"])
    X = df.drop(columns=["disease_label"])

    print("Features used for training:")
    for f in X.columns:
        print(f"  • {f}")
    print()

    return X, y, target_le, encoders


# ─────────────────────────────────────────────
# 3. TRAIN / EVALUATE
# ─────────────────────────────────────────────

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy : {acc:.4f}\n")

    return model, X_train, X_test, y_train, y_test, y_pred


# ─────────────────────────────────────────────
# 4. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, target_le):
    cm = confusion_matrix(y_test, y_pred)
    labels = target_le.classes_

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlOrBr",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5
    )
    plt.title("Confusion Matrix — Crop Disease Detector", fontsize=14, pad=12)
    plt.ylabel("Actual", fontsize=11)
    plt.xlabel("Predicted", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals  = importances[indices]

    plt.figure(figsize=(9, 5))
    colors = plt.cm.YlOrBr(np.linspace(0.3, 0.85, len(sorted_names)))
    bars = plt.bar(sorted_names, sorted_vals, color=colors, edgecolor="white")
    plt.title("Feature Importance — Random Forest", fontsize=14, pad=10)
    plt.ylabel("Importance Score", fontsize=11)
    plt.xlabel("Feature", fontsize=11)
    plt.xticks(rotation=25, ha="right")
    for bar, val in zip(bars, sorted_vals):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.show()
    print("Saved: feature_importance.png")


def plot_class_distribution(df):
    counts = df['disease_label'].value_counts()
    colors = sns.color_palette("YlOrBr", len(counts))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart
    axes[0].barh(counts.index, counts.values, color=colors)
    axes[0].set_title("Samples per Disease Class", fontsize=12)
    axes[0].set_xlabel("Count")
    for i, v in enumerate(counts.values):
        axes[0].text(v + 0.3, i, str(v), va="center", fontsize=9)

    # Pie chart
    axes[1].pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1}
    )
    axes[1].set_title("Disease Class Distribution", fontsize=12)

    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=150)
    plt.show()
    print("Saved: class_distribution.png")


def plot_accuracy_vs_estimators(X, y):
    estimator_range = [10, 20, 50, 100, 150, 200]
    accuracies = []

    for n in estimator_range:
        clf = RandomForestClassifier(n_estimators=n, random_state=42,
                                     class_weight="balanced")
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        accuracies.append(scores.mean())

    plt.figure(figsize=(8, 4))
    plt.plot(estimator_range, accuracies, marker="o",
             color="#d95f02", linewidth=2, markersize=7)
    plt.fill_between(estimator_range, accuracies, alpha=0.15, color="#d95f02")
    plt.title("Model Accuracy vs Number of Trees", fontsize=13)
    plt.xlabel("Number of Estimators (Trees)", fontsize=11)
    plt.ylabel("CV Accuracy", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("accuracy_vs_estimators.png", dpi=150)
    plt.show()
    print("Saved: accuracy_vs_estimators.png")


# ─────────────────────────────────────────────
# 5. CLASSIFICATION REPORT
# ─────────────────────────────────────────────

def print_report(y_test, y_pred, target_le):
    print("\n" + "=" * 55)
    print("  CLASSIFICATION REPORT")
    print("=" * 55)
    print(classification_report(
        y_test, y_pred,
        target_names=target_le.classes_
    ))


# ─────────────────────────────────────────────
# 6. PREDICT — SINGLE SAMPLE
# ─────────────────────────────────────────────

def predict_single(model, encoders, target_le, feature_names, sample: dict):
    """
    Predict disease for a single observation.

    sample = {
        'crop_type': 'tomato',
        'leaf_color': 'yellow',
        'spot_size': 'medium',
        'spot_color': 'brown',
        'yellowing': 'yes',
        'wilting': 'no',
        'lesion_count': 8,
        'humidity': 80,
        'temperature': 32
    }
    """
    row = {}
    cat_cols = ["crop_type", "leaf_color", "spot_size",
                "spot_color", "yellowing", "wilting"]

    for col in feature_names:
        val = sample[col]
        if col in cat_cols:
            le = encoders[col]
            val_str = str(val)
            if val_str not in le.classes_:
                print(f"Warning: '{val_str}' not seen in training for '{col}'.")
                val = 0
            else:
                val = le.transform([val_str])[0]
        row[col] = val

    X_input = pd.DataFrame([row])
    pred_encoded = model.predict(X_input)[0]
    pred_proba = model.predict_proba(X_input)[0]
    pred_label = target_le.inverse_transform([pred_encoded])[0]
    confidence = pred_proba.max() * 100

    print("\n" + "─" * 45)
    print(f"  Input crop    : {sample['crop_type']}")
    print(f"  Predicted     : {pred_label}")
    print(f"  Confidence    : {confidence:.1f}%")
    print("─" * 45)

    print("\nProbability breakdown:")
    for cls, prob in sorted(
        zip(target_le.classes_, pred_proba),
        key=lambda x: -x[1]
    ):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<30} {bar} {prob*100:5.1f}%")

    return pred_label, confidence


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # --- Load ---
    df = load_data("crop_disease_dataset.csv")

    # --- Visualise distribution ---
    plot_class_distribution(df)

    # --- Preprocess ---
    X, y, target_le, encoders = preprocess(df)

    # --- Train & evaluate ---
    model, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)

    # --- Report ---
    print_report(y_test, y_pred, target_le)

    # --- Plots ---
    plot_confusion_matrix(y_test, y_pred, target_le)
    plot_feature_importance(model, list(X.columns))
    plot_accuracy_vs_estimators(X, y)

    # --- Demo prediction ---
    print("\n=== DEMO PREDICTION ===")
    sample = {
        "crop_type": "tomato",
        "leaf_color": "yellow",
        "spot_size": "medium",
        "spot_color": "brown",
        "yellowing": "yes",
        "wilting": "no",
        "lesion_count": 8,
        "humidity": 80,
        "temperature": 32
    }
    predict_single(model, encoders, target_le, list(X.columns), sample)

    print("\n✓ All outputs saved. Project complete.")
