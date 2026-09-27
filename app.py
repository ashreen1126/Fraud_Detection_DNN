import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Fraud Detection System",
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
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


# ============================================================
# LOAD FEATURES
# ============================================================

@st.cache_data
def load_features():
    return joblib.load(FEATURES_PATH)


# ============================================================
# LOAD METRICS
# ============================================================

@st.cache_data
def load_metrics():
    return joblib.load(METRICS_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

required_files = [
    MODEL_PATH,
    SCALER_PATH,
    FEATURES_PATH,
    METRICS_PATH,
    DATA_PATH
]

missing_files = [
    str(file)
    for file in required_files
    if not file.exists()
]

if missing_files:

    st.error("Some required project files are missing:")

    for file in missing_files:
        st.write(file)

    st.stop()


# ============================================================
# LOAD ALL PROJECT COMPONENTS
# ============================================================

model = load_model()
scaler = load_scaler()
features = load_features()
metrics = load_metrics()
df = load_dataset()


# ============================================================
# HANDLE METRIC NAMES SAFELY
# ============================================================

accuracy_value = metrics.get("accuracy", 0)

precision_value = metrics.get("precision", 0)

recall_value = metrics.get("recall", 0)

# FIXED F1 SCORE
f1_value = metrics.get(
    "f1",
    metrics.get(
        "f1_score",
        0
    )
)

roc_auc_value = metrics.get("roc_auc", 0)

pr_auc_value = metrics.get(
    "pr_auc",
    metrics.get(
        "pr_auc_score",
        0
    )
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔐 Fraud Detection")

st.sidebar.write(
    "AI-Powered Deep Neural Network "
    "for Fraudulent Transaction Detection"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Transaction Prediction",
        "Model Performance",
        "About Model"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Model: Deep Neural Network\n\n"
    "Activation: ReLU + Sigmoid\n\n"
    "Optimizer: Adam\n\n"
    "Dataset: Credit Card Transactions"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title(
        "🔐 AI-Powered Fraudulent Transaction Detection System"
    )

    st.markdown(
        """
        ### Welcome

        This system uses a **Deep Neural Network (DNN)** to identify
        potentially fraudulent credit card transactions.

        The model uses **ReLU activation functions** in the hidden
        layers and a **Sigmoid activation function** in the output
        layer.
        """
    )

    st.markdown("---")


    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    total_transactions = len(df)

    fraud_count = int(
        df["Class"].sum()
    )

    genuine_count = (
        total_transactions -
        fraud_count
    )

    fraud_rate = (
        fraud_count /
        total_transactions
    ) * 100


    # ========================================================
    # DATASET METRICS
    # ========================================================

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


    st.markdown("---")


    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    st.subheader("Model Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy_value:.4f}"
        )

    with col2:

        st.metric(
            "Precision",
            f"{precision_value:.4f}"
        )

    with col3:

        st.metric(
            "Recall",
            f"{recall_value:.4f}"
        )

    with col4:

        st.metric(
            "F1 Score",
            f"{f1_value:.4f}"
        )

    with col5:

        st.metric(
            "ROC-AUC",
            f"{roc_auc_value:.4f}"
        )


    st.markdown("---")


    # ========================================================
    # TRANSACTION DISTRIBUTION
    # ========================================================

    st.subheader(
        "Transaction Distribution"
    )

    class_data = pd.DataFrame(
        {
            "Transaction Type": [
                "Genuine",
                "Fraudulent"
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
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "The dataset is highly imbalanced because fraudulent "
        "transactions represent only a small portion of all "
        "transactions."
    )


# ============================================================
# TRANSACTION PREDICTION
# ============================================================

elif page == "Transaction Prediction":

    st.title(
        "🔍 Transaction Fraud Prediction"
    )

    st.write(
        "Upload a transaction CSV file or use one of the sample "
        "transactions to test the trained DNN model."
    )

    st.markdown("---")


    # ========================================================
    # QUICK DEMO
    # ========================================================

    st.subheader("Quick Demo")

    col1, col2 = st.columns(2)


    # ========================================================
    # GENUINE SAMPLE
    # ========================================================

    with col1:

        if st.button(
            "Test Sample Genuine Transaction",
            use_container_width=True
        ):

            genuine_rows = df[
                df["Class"] == 0
            ]

            if len(genuine_rows) > 0:

                sample = genuine_rows.iloc[0]

                X_sample = (
                    sample[features]
                    .values
                    .reshape(1, -1)
                )

                X_scaled = scaler.transform(
                    X_sample
                )

                probability = float(
                    model.predict(
                        X_scaled,
                        verbose=0
                    )[0][0]
                )

                st.session_state[
                    "prediction_probability"
                ] = probability

                st.session_state[
                    "prediction_type"
                ] = "Demo Genuine"


    # ========================================================
    # FRAUD SAMPLE
    # ========================================================

    with col2:

        if st.button(
            "Test Sample Fraud Transaction",
            use_container_width=True
        ):

            fraud_rows = df[
                df["Class"] == 1
            ]

            if len(fraud_rows) > 0:

                sample = fraud_rows.iloc[0]

                X_sample = (
                    sample[features]
                    .values
                    .reshape(1, -1)
                )

                X_scaled = scaler.transform(
                    X_sample
                )

                probability = float(
                    model.predict(
                        X_scaled,
                        verbose=0
                    )[0][0]
                )

                st.session_state[
                    "prediction_probability"
                ] = probability

                st.session_state[
                    "prediction_type"
                ] = "Demo Fraud"


    # ========================================================
    # DISPLAY DEMO RESULT
    # ========================================================

    if "prediction_probability" in st.session_state:

        probability = (
            st.session_state[
                "prediction_probability"
            ]
        )

        prediction_type = (
            st.session_state[
                "prediction_type"
            ]
        )

        prediction = (
            1
            if probability >= 0.5
            else 0
        )

        risk_percentage = (
            probability * 100
        )


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        if risk_percentage < 30:

            risk_level = "Low Risk"

        elif risk_percentage < 70:

            risk_level = "Medium Risk"

        else:

            risk_level = "High Risk"


        st.markdown("---")

        st.subheader(
            "Prediction Result"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Fraud Probability",
                f"{risk_percentage:.2f}%"
            )


        with col2:

            if prediction == 1:

                st.error(
                    "🚨 FRAUDULENT TRANSACTION"
                )

            else:

                st.success(
                    "✅ GENUINE TRANSACTION"
                )


        with col3:

            st.metric(
                "Risk Level",
                risk_level
            )


        st.caption(
            f"Prediction source: {prediction_type}"
        )


    # ========================================================
    # CSV UPLOAD
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Upload Transaction CSV"
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV containing transaction features",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(
                uploaded_file
            )


            st.write(
                "Uploaded data:"
            )

            st.dataframe(
                uploaded_df.head(),
                use_container_width=True
            )


            # ------------------------------------------------
            # COPY DATA
            # ------------------------------------------------

            prediction_df = (
                uploaded_df.copy()
            )


            # ------------------------------------------------
            # REMOVE CLASS COLUMN
            # ------------------------------------------------

            if "Class" in prediction_df.columns:

                prediction_df = (
                    prediction_df.drop(
                        columns=["Class"]
                    )
                )


            # ------------------------------------------------
            # CHECK FEATURES
            # ------------------------------------------------

            missing_features = [
                feature
                for feature in features
                if feature not in prediction_df.columns
            ]


            if missing_features:

                st.error(
                    "The uploaded CSV is missing required features:"
                )

                st.write(
                    missing_features
                )


            else:

                # --------------------------------------------
                # SELECT FEATURES
                # --------------------------------------------

                prediction_df = (
                    prediction_df[features]
                )


                # --------------------------------------------
                # SCALE DATA
                # --------------------------------------------

                X_scaled = scaler.transform(
                    prediction_df
                )


                # --------------------------------------------
                # MODEL PREDICTION
                # --------------------------------------------

                probabilities = (
                    model.predict(
                        X_scaled,
                        verbose=0
                    )
                    .flatten()
                )


                # --------------------------------------------
                # CREATE RESULTS
                # --------------------------------------------

                results = uploaded_df.copy()


                results[
                    "Fraud Probability"
                ] = probabilities


                results[
                    "Prediction"
                ] = np.where(
                    probabilities >= 0.5,
                    "Fraudulent",
                    "Genuine"
                )


                # --------------------------------------------
                # RISK LEVEL
                # --------------------------------------------

                results[
                    "Risk Level"
                ] = pd.cut(
                    probabilities,
                    bins=[
                        -0.01,
                        0.30,
                        0.70,
                        1.0
                    ],
                    labels=[
                        "Low Risk",
                        "Medium Risk",
                        "High Risk"
                    ]
                )


                # --------------------------------------------
                # SHOW RESULTS
                # --------------------------------------------

                st.subheader(
                    "Prediction Results"
                )

                st.dataframe(
                    results,
                    use_container_width=True
                )


                # --------------------------------------------
                # DOWNLOAD RESULTS
                # --------------------------------------------

                csv_data = (
                    results
                    .to_csv(index=False)
                    .encode("utf-8")
                )


                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=csv_data,
                    file_name="fraud_predictions.csv",
                    mime="text/csv"
                )


                # --------------------------------------------
                # SUMMARY
                # --------------------------------------------

                fraud_predictions = int(
                    (
                        probabilities >= 0.5
                    ).sum()
                )


                genuine_predictions = (
                    len(probabilities)
                    -
                    fraud_predictions
                )


                col1, col2 = st.columns(2)


                with col1:

                    st.metric(
                        "Predicted Genuine",
                        genuine_predictions
                    )


                with col2:

                    st.metric(
                        "Predicted Fraud",
                        fraud_predictions
                    )


        except Exception as e:

            st.error(
                f"Unable to process the uploaded file: {e}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title(
        "📊 Model Performance"
    )

    st.write(
        "Performance metrics and evaluation plots generated "
        "during model testing."
    )

    st.markdown("---")


    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Accuracy",
            f"{accuracy_value:.4f}"
        )

        st.metric(
            "Precision",
            f"{precision_value:.4f}"
        )


    with col2:

        st.metric(
            "Recall",
            f"{recall_value:.4f}"
        )

        st.metric(
            "F1 Score",
            f"{f1_value:.4f}"
        )


    with col3:

        st.metric(
            "ROC-AUC",
            f"{roc_auc_value:.4f}"
        )

        st.metric(
            "PR-AUC",
            f"{pr_auc_value:.4f}"
        )


    st.markdown("---")


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "Confusion Matrix"
    )

    confusion_path = (
        BASE_DIR /
        "plots" /
        "confusion_matrix.png"
    )


    if confusion_path.exists():

        st.image(
            str(confusion_path),
            use_container_width=True
        )

    else:

        st.warning(
            "Confusion matrix image not found."
        )


    # ========================================================
    # ROC CURVE
    # ========================================================

    st.subheader(
        "ROC Curve"
    )

    roc_path = (
        BASE_DIR /
        "plots" /
        "roc_curve.png"
    )


    if roc_path.exists():

        st.image(
            str(roc_path),
            use_container_width=True
        )

    else:

        st.warning(
            "ROC curve image not found."
        )


    # ========================================================
    # PRECISION-RECALL CURVE
    # ========================================================

    st.subheader(
        "Precision-Recall Curve"
    )

    pr_path = (
        BASE_DIR /
        "plots" /
        "precision_recall_curve.png"
    )


    if pr_path.exists():

        st.image(
            str(pr_path),
            use_container_width=True
        )

    else:

        st.warning(
            "Precision-Recall curve image not found."
        )


    # ========================================================
    # TRAINING LOSS
    # ========================================================

    st.subheader(
        "Training Loss"
    )

    loss_path = (
        BASE_DIR /
        "plots" /
        "training_loss.png"
    )


    if loss_path.exists():

        st.image(
            str(loss_path),
            use_container_width=True
        )

    else:

        st.warning(
            "Training loss image not found."
        )


    # ========================================================
    # TRAINING ACCURACY
    # ========================================================

    st.subheader(
        "Training Accuracy"
    )

    accuracy_path = (
        BASE_DIR /
        "plots" /
        "training_accuracy.png"
    )


    if accuracy_path.exists():

        st.image(
            str(accuracy_path),
            use_container_width=True
        )

    else:

        st.warning(
            "Training accuracy image not found."
        )


# ============================================================
# ABOUT MODEL
# ============================================================

elif page == "About Model":

    st.title(
        "🧠 About the DNN Model"
    )


    st.markdown(
        """
        ## Deep Neural Network Architecture

        The fraud detection system uses a Deep Neural Network.

        ### Architecture

        **Input Layer**
        - Transaction features
        - 30 input features

        **Hidden Layer 1**
        - 64 neurons
        - ReLU activation

        **Hidden Layer 2**
        - 32 neurons
        - ReLU activation

        **Output Layer**
        - 1 neuron
        - Sigmoid activation


        ### Why ReLU?

        ReLU helps the neural network learn non-linear patterns
        efficiently in the hidden layers.


        ### Why Sigmoid?

        The final Sigmoid function produces a value between 0 and 1.
        This value represents the model's estimated probability
        of fraud.


        ### Handling Class Imbalance

        The dataset contains far fewer fraudulent transactions
        than genuine transactions.

        Therefore, class weights were used during training so that
        fraudulent transactions receive greater importance.


        ### Model Training

        - Optimizer: Adam
        - Loss Function: Binary Crossentropy
        - Batch Size: 256
        - Early Stopping: Enabled


        ### Prediction

        The model produces a fraud probability.

        A probability of 0.5 or above is classified as fraudulent
        in this prototype.
        """
    )


    st.markdown("---")


    # ========================================================
    # PROJECT WORKFLOW
    # ========================================================

    st.subheader(
        "Project Workflow"
    )


    st.code(
        """
Transaction Data
       ↓
Data Cleaning
       ↓
Train/Test Split
       ↓
Feature Scaling
       ↓
Class Weighting
       ↓
Deep Neural Network
       ↓
ReLU Hidden Layers
       ↓
Sigmoid Output
       ↓
Fraud Probability
       ↓
Fraud / Genuine Prediction
       ↓
Dashboard & Report
        """,
        language="text"
    )


    st.success(
        "The prototype is connected to the trained DNN model "
        "and the real credit card transaction dataset."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Deep Neural Network for Fraudulent Transaction Detection | "
    "AI & Data Science Project"
)