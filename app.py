import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Detection DNN",
    page_icon="🔐",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "fraud_model.keras"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"
FEATURES_PATH = BASE_DIR / "models" / "features.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.joblib"

PLOTS_DIR = BASE_DIR / "plots"


# ============================================================
# LOAD MODEL COMPONENTS
# ============================================================

@st.cache_resource
def load_fraud_model():
    return load_model(MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_resource
def load_features():
    return joblib.load(FEATURES_PATH)


@st.cache_resource
def load_metrics():
    return joblib.load(METRICS_PATH)


# ============================================================
# CHECK REQUIRED MODEL FILES
# ============================================================

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH,
    METRICS_PATH
]

missing_files = [
    str(file)
    for file in required_files
    if not file.exists()
]

if missing_files:

    st.error("Some required model files are missing:")

    for file in missing_files:
        st.write(file)

    st.stop()


# ============================================================
# LOAD PROJECT COMPONENTS
# ============================================================

model = load_fraud_model()
scaler = load_scaler()
features = load_features()
metrics = load_metrics()


# ============================================================
# DATASET STATISTICS
# ============================================================

total_transactions = 284807
fraud_count = 492
genuine_count = 284315

fraud_rate = (
    fraud_count / total_transactions
) * 100


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def predict_transaction(input_data):

    input_data = np.asarray(input_data)

    if input_data.ndim == 1:
        input_data = input_data.reshape(1, -1)

    input_scaled = scaler.transform(input_data)

    probability = float(
        model.predict(
            input_scaled,
            verbose=0
        )[0][0]
    )

    prediction = (
        "Fraudulent Transaction"
        if probability >= 0.5
        else "Genuine Transaction"
    )

    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.70:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return probability, prediction, risk


# ============================================================
# TITLE
# ============================================================

st.title("🔐 Deep Neural Network for Fraud Detection")

st.markdown(
    """
    **AI-powered fraudulent transaction detection using a
    Deep Neural Network with ReLU and Sigmoid activation functions.**
    """
)

st.divider()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Transaction Prediction",
        "Model Performance",
        "About Model"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Fraud Detection Dashboard")

    st.write(
        "Overview of the financial transaction dataset and "
        "the trained fraud detection model."
    )

    # --------------------------------------------------------
    # DATASET METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Transactions",
            f"{total_transactions:,}"
        )

    with col2:
        st.metric(
            "Genuine Transactions",
            f"{genuine_count:,}"
        )

    with col3:
        st.metric(
            "Fraudulent Transactions",
            f"{fraud_count:,}"
        )

    with col4:
        st.metric(
            "Fraud Rate",
            f"{fraud_rate:.3f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    st.subheader("Model Performance")

    accuracy = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)

    f1_value = metrics.get(
        "f1",
        metrics.get("f1_score", 0)
    )

    roc_auc = metrics.get("roc_auc", 0)

    pr_auc = metrics.get(
        "pr_auc",
        metrics.get("pr_auc_score", 0)
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{f1_value * 100:.2f}%"
        )

    with col5:
        st.metric(
            "ROC-AUC",
            f"{roc_auc * 100:.2f}%"
        )

    with col6:
        st.metric(
            "PR-AUC",
            f"{pr_auc * 100:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Transaction Class Distribution")

    class_data = pd.DataFrame(
        {
            "Transaction Type": [
                "Genuine",
                "Fraud"
            ],
            "Count": [
                genuine_count,
                fraud_count
            ]
        }
    )

    fig = px.bar(
        class_data,
        x="Transaction Type",
        y="Count",
        title="Genuine vs Fraudulent Transactions",
        text="Count"
    )

    fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "The dataset is highly imbalanced because fraudulent "
        "transactions represent only a small fraction of all transactions."
    )


# ============================================================
# TRANSACTION PREDICTION
# ============================================================

elif page == "Transaction Prediction":

    st.header("🔎 Transaction Prediction")

    st.write(
        "Use the trained Deep Neural Network to classify a transaction."
    )

    # --------------------------------------------------------
    # DEMO BUTTONS
    # --------------------------------------------------------

    st.subheader("Quick Demo")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "Test Sample Genuine Transaction",
            use_container_width=True
        ):

            sample = np.zeros(
                len(features)
            )

            X_sample = sample.reshape(
                1,
                -1
            )

            probability, prediction, risk = predict_transaction(
                X_sample
            )

            st.session_state[
                "prediction_probability"
            ] = probability

            st.session_state[
                "prediction_result"
            ] = prediction

            st.session_state[
                "prediction_risk"
            ] = risk

            st.session_state[
                "prediction_type"
            ] = "Demo Genuine"

    with col2:

        if st.button(
            "Test Sample Fraud Transaction",
            use_container_width=True
        ):

            sample = np.zeros(
                len(features)
            )

            X_sample = sample.reshape(
                1,
                -1
            )

            probability, prediction, risk = predict_transaction(
                X_sample
            )

            st.session_state[
                "prediction_probability"
            ] = probability

            st.session_state[
                "prediction_result"
            ] = prediction

            st.session_state[
                "prediction_risk"
            ] = risk

            st.session_state[
                "prediction_type"
            ] = "Demo Fraud"

    # --------------------------------------------------------
    # DISPLAY DEMO RESULT
    # --------------------------------------------------------

    if "prediction_probability" in st.session_state:

        probability = st.session_state[
            "prediction_probability"
        ]

        prediction = st.session_state[
            "prediction_result"
        ]

        risk = st.session_state[
            "prediction_risk"
        ]

        prediction_type = st.session_state.get(
            "prediction_type",
            "Prediction"
        )

        st.divider()

        st.subheader(
            f"{prediction_type} Result"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Fraud Probability",
                f"{probability * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Prediction",
                prediction
            )

        with col3:

            st.metric(
                "Risk Level",
                risk
            )

        if prediction == "Fraudulent Transaction":

            st.error(
                "⚠️ This transaction has been classified as potentially fraudulent."
            )

        else:

            st.success(
                "✅ This transaction has been classified as genuine."
            )

    st.divider()

    # --------------------------------------------------------
    # CSV UPLOAD
    # --------------------------------------------------------

    st.subheader("Upload Transaction CSV")

    st.write(
        "Upload a CSV file containing the same input features "
        "used during model training."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )

            st.write("Uploaded Data")

            st.dataframe(
                uploaded_df.head(),
                use_container_width=True
            )

            missing_features = [
                feature
                for feature in features
                if feature not in uploaded_df.columns
            ]

            if missing_features:

                st.error(
                    "The uploaded CSV is missing required features:"
                )

                st.write(
                    missing_features
                )

            else:

                input_data = uploaded_df[
                    features
                ]

                probabilities = model.predict(
                    scaler.transform(input_data),
                    verbose=0
                ).flatten()

                results_df = uploaded_df.copy()

                results_df[
                    "Fraud Probability"
                ] = probabilities

                results_df[
                    "Prediction"
                ] = np.where(
                    probabilities >= 0.5,
                    "Fraudulent Transaction",
                    "Genuine Transaction"
                )

                results_df[
                    "Risk Level"
                ] = pd.cut(
                    probabilities,
                    bins=[
                        -np.inf,
                        0.30,
                        0.70,
                        np.inf
                    ],
                    labels=[
                        "Low Risk",
                        "Medium Risk",
                        "High Risk"
                    ]
                )

                st.subheader(
                    "Prediction Results"
                )

                st.dataframe(
                    results_df,
                    use_container_width=True
                )

                csv_data = results_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Prediction Results",
                    data=csv_data,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(
                f"Error processing uploaded file: {e}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("📈 Model Performance")

    st.write(
        "Performance evaluation of the trained Deep Neural Network."
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = metrics.get("accuracy", 0)
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)

    f1_value = metrics.get(
        "f1",
        metrics.get("f1_score", 0)
    )

    roc_auc = metrics.get("roc_auc", 0)

    pr_auc = metrics.get(
        "pr_auc",
        metrics.get("pr_auc_score", 0)
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        st.metric(
            "Precision",
            f"{precision * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Recall",
            f"{recall * 100:.2f}%"
        )

        st.metric(
            "F1 Score",
            f"{f1_value * 100:.2f}%"
        )

    with col3:

        st.metric(
            "ROC-AUC",
            f"{roc_auc * 100:.2f}%"
        )

        st.metric(
            "PR-AUC",
            f"{pr_auc * 100:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    confusion_matrix_path = (
        PLOTS_DIR / "confusion_matrix.png"
    )

    if confusion_matrix_path.exists():

        st.subheader("Confusion Matrix")

        st.image(
            str(confusion_matrix_path),
            use_container_width=True
        )

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    roc_curve_path = (
        PLOTS_DIR / "roc_curve.png"
    )

    if roc_curve_path.exists():

        st.subheader("ROC Curve")

        st.image(
            str(roc_curve_path),
            use_container_width=True
        )

    # --------------------------------------------------------
    # PRECISION-RECALL CURVE
    # --------------------------------------------------------

    pr_curve_path = (
        PLOTS_DIR / "precision_recall_curve.png"
    )

    if pr_curve_path.exists():

        st.subheader(
            "Precision-Recall Curve"
        )

        st.image(
            str(pr_curve_path),
            use_container_width=True
        )

    # --------------------------------------------------------
    # TRAINING LOSS
    # --------------------------------------------------------

    training_loss_path = (
        PLOTS_DIR / "training_loss.png"
    )

    if training_loss_path.exists():

        st.subheader(
            "Training Loss"
        )

        st.image(
            str(training_loss_path),
            use_container_width=True
        )

    # --------------------------------------------------------
    # TRAINING ACCURACY
    # --------------------------------------------------------

    training_accuracy_path = (
        PLOTS_DIR / "training_accuracy.png"
    )

    if training_accuracy_path.exists():

        st.subheader(
            "Training Accuracy"
        )

        st.image(
            str(training_accuracy_path),
            use_container_width=True
        )


# ============================================================
# ABOUT MODEL
# ============================================================

elif page == "About Model":

    st.header("🧠 About the Model")

    st.subheader(
        "Deep Neural Network Architecture"
    )

    st.code(
        """
Input Features
      ↓
Dense Layer – 64 Neurons
ReLU Activation
      ↓
Dense Layer – 32 Neurons
ReLU Activation
      ↓
Output Layer – 1 Neuron
Sigmoid Activation
      ↓
Fraud Probability
        """,
        language="text"
    )

    st.divider()

    # --------------------------------------------------------
    # RELU
    # --------------------------------------------------------

    st.subheader(
        "ReLU Activation"
    )

    st.write(
        "ReLU (Rectified Linear Unit) is used in the hidden "
        "layers to introduce non-linearity into the neural network."
    )

    st.latex(
        r"ReLU(x) = max(0,x)"
    )

    # --------------------------------------------------------
    # SIGMOID
    # --------------------------------------------------------

    st.subheader(
        "Sigmoid Activation"
    )

    st.write(
        "Sigmoid is used in the output layer to produce a "
        "probability between 0 and 1."
    )

    st.latex(
        r"\sigma(x) = \frac{1}{1+e^{-x}}"
    )

    # --------------------------------------------------------
    # CLASS IMBALANCE
    # --------------------------------------------------------

    st.subheader(
        "Handling Class Imbalance"
    )

    st.write(
        f"""
        The dataset contains {total_transactions:,} transactions,
        including {genuine_count:,} genuine transactions and
        {fraud_count:,} fraudulent transactions.

        Because fraud cases represent only a small percentage of
        the dataset, balanced class weights were used during training
        to give greater importance to the minority fraud class.
        """
    )

    # --------------------------------------------------------
    # TRAINING CONFIGURATION
    # --------------------------------------------------------

    st.subheader(
        "Training Configuration"
    )

    training_details = pd.DataFrame(
        {
            "Parameter": [
                "Optimizer",
                "Loss Function",
                "Batch Size",
                "Maximum Epochs",
                "Hidden Layers",
                "Activation",
                "Output Activation",
                "Classification Threshold"
            ],
            "Value": [
                "Adam",
                "Binary Crossentropy",
                "256",
                "30",
                "64 → 32",
                "ReLU",
                "Sigmoid",
                "0.5"
            ]
        }
    )

    st.dataframe(
        training_details,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.subheader(
        "Project Workflow"
    )

    st.code(
        """
Financial Transaction Data
          ↓
Data Preprocessing
          ↓
Train / Test Split
          ↓
Feature Scaling
          ↓
Class Weight Calculation
          ↓
Deep Neural Network
          ↓
ReLU Hidden Layers
          ↓
Sigmoid Output
          ↓
Fraud Probability
          ↓
Fraud / Genuine Classification
          ↓
Risk Level
        """,
        language="text"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Deep Neural Network for Fraudulent Transaction Detection | "
    "AI & Data Science Project"
)