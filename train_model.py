import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# --------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------

DATA_PATH = "data/creditcard.csv"
MODEL_DIR = "models"
PLOT_DIR = "plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# 3. BASIC DATA INFORMATION
# --------------------------------------------------

print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution:")
print(df["Class"].value_counts())

fraud_count = (df["Class"] == 1).sum()
genuine_count = (df["Class"] == 0).sum()

print("\nGenuine transactions:", genuine_count)
print("Fraudulent transactions:", fraud_count)

fraud_percentage = fraud_count / len(df) * 100

print(f"Fraud percentage: {fraud_percentage:.4f}%")

# --------------------------------------------------
# 4. CHECK MISSING VALUES
# --------------------------------------------------

print("\nMissing values:")

missing_values = df.isnull().sum().sum()

if missing_values == 0:
    print("No missing values found.")
else:
    print("Missing values found:", missing_values)

# --------------------------------------------------
# 5. REMOVE DUPLICATES
# --------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("Duplicates removed.")

# --------------------------------------------------
# 6. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

feature_names = X.columns.tolist()

print("\nNumber of features:", len(feature_names))

# --------------------------------------------------
# 7. TRAIN / TEST SPLIT
# --------------------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 8. FEATURE SCALING
# --------------------------------------------------

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")

# Save feature names
joblib.dump(feature_names, f"{MODEL_DIR}/features.joblib")

print("Scaler saved.")

# --------------------------------------------------
# 9. HANDLE CLASS IMBALANCE
# --------------------------------------------------

print("\nCalculating class weights...")

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("Class weights:")
print(class_weights)

# --------------------------------------------------
# 10. BUILD DEEP NEURAL NETWORK
# --------------------------------------------------

print("\nBuilding Deep Neural Network...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")

model.summary()

# --------------------------------------------------
# 11. EARLY STOPPING
# --------------------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# --------------------------------------------------
# 12. TRAIN MODEL
# --------------------------------------------------

print("\nStarting model training...")

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=30,
    batch_size=256,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

print("\nTraining completed!")

# --------------------------------------------------
# 13. SAVE MODEL
# --------------------------------------------------

model_path = f"{MODEL_DIR}/fraud_model.keras"

model.save(model_path)

print("\nModel saved to:")
print(model_path)

# --------------------------------------------------
# 14. PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

y_probability = model.predict(
    X_test_scaled,
    verbose=0
).ravel()

threshold = 0.50

y_prediction = (
    y_probability >= threshold
).astype(int)

# --------------------------------------------------
# 15. PERFORMANCE METRICS
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_prediction
)

precision = precision_score(
    y_test,
    y_prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

# --------------------------------------------------
# 16. SAVE METRICS
# --------------------------------------------------

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "threshold": threshold
}

joblib.dump(
    metrics,
    f"{MODEL_DIR}/metrics.joblib"
)

# --------------------------------------------------
# 17. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_prediction
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Genuine", "Fraud"],
    yticklabels=["Genuine", "Fraud"]
)

plt.title("Fraud Detection Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/confusion_matrix.png",
    dpi=200
)

plt.close()

# --------------------------------------------------
# 18. ROC CURVE
# --------------------------------------------------

fpr, tpr, _ = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/roc_curve.png",
    dpi=200
)

plt.close()

# --------------------------------------------------
# 19. PRECISION-RECALL CURVE
# --------------------------------------------------

precision_values, recall_values, _ = precision_recall_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_values,
    precision_values,
    label=f"PR-AUC = {pr_auc:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/precision_recall_curve.png",
    dpi=200
)

plt.close()

# --------------------------------------------------
# 20. TRAINING LOSS
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training and Validation Loss")

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/training_loss.png",
    dpi=200
)

plt.close()

# --------------------------------------------------
# 21. TRAINING ACCURACY
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Training and Validation Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/training_accuracy.png",
    dpi=200
)

plt.close()

# --------------------------------------------------
# 22. CLASS DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(7, 5))

df["Class"].value_counts().sort_index().plot(
    kind="bar"
)

plt.xticks(
    [0, 1],
    ["Genuine", "Fraud"],
    rotation=0
)

plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")

plt.title("Class Distribution")

plt.tight_layout()

plt.savefig(
    f"{PLOT_DIR}/class_distribution.png",
    dpi=200
)

plt.close()

# --------------------------------------------------
# 23. FINAL SUMMARY
# --------------------------------------------------

print("\n========================================")
print("PROJECT TRAINING COMPLETE")
print("========================================")

print("Model      :", model_path)
print("Scaler     :", f"{MODEL_DIR}/scaler.joblib")
print("Features   :", f"{MODEL_DIR}/features.joblib")
print("Metrics    :", f"{MODEL_DIR}/metrics.joblib")
print("Plots saved:", PLOT_DIR)

print("\nThe DNN uses:")
print("Hidden Layer 1 : 64 neurons + ReLU")
print("Hidden Layer 2 : 32 neurons + ReLU")
print("Output Layer   : 1 neuron + Sigmoid")

print("\nFraud detection threshold:", threshold)

print("\nReady for Streamlit application!")