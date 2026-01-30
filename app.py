import streamlit as st
import os
import joblib
import json
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input

# -----------------------------
# Base directory (VERY IMPORTANT)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# -----------------------------
# Load model safely
# -----------------------------
@st.cache_resource
def load_selected_model(model_choice):
    """
    Loads the selected model safely and returns it with any extra objects (PCA or class names).
    """
    try:
        if model_choice == "Logistic Regression":
            model_path = os.path.join(MODELS_DIR, "logistic.pkl")
            pca_path = os.path.join(BASE_DIR, "pca.pkl")
            model = joblib.load(model_path)
            pca = joblib.load(pca_path)
            return model, pca

        elif model_choice == "Random Forest":
            model_path = os.path.join(MODELS_DIR, "rf_model.pkl")
            pca_path = os.path.join(BASE_DIR, "pca.pkl")
            model = joblib.load(model_path)
            pca = joblib.load(pca_path)
            return model, pca

        elif model_choice == "CNN":
            model_path = os.path.join(MODELS_DIR, "garbage_cnn_model.h5")
            class_path = os.path.join(MODELS_DIR, "class_names.json")
            model = tf.keras.models.load_model(model_path)
            if not os.path.exists(class_path):
                st.error(f"Class names file not found: {class_path}")
                return model, []
            with open(class_path, "r") as f:
                class_names = json.load(f)
            return model, class_names

        elif model_choice == "VGG16":
            model_path = os.path.join(MODELS_DIR, "garbage_vgg16.h5")
            class_path = os.path.join(MODELS_DIR, "class_names.json")
            model = tf.keras.models.load_model(model_path)
            if not os.path.exists(class_path):
                st.error(f"Class names file not found: {class_path}")
                return model, []
            with open(class_path, "r") as f:
                class_names = json.load(f)
            return model, class_names

        else:
            st.error("Invalid model choice!")
            return None, None

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# -----------------------------
# Load class names safely (global for ML/CNN)
# -----------------------------
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
if not os.path.exists(CLASS_NAMES_PATH):
    st.warning(f"Warning: Class names file not found: {CLASS_NAMES_PATH}")
    class_names = []
else:
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
NUM_CLASSES = len(class_names)

# -----------------------------
# Preprocessing functions
# -----------------------------
ML_IMG_SIZE = 32        # For Logistic Regression / Random Forest
DL_IMG_SIZE = 128       # For CNN / VGG16

def preprocess_ml(image):
    """For Logistic Regression and Random Forest"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (ML_IMG_SIZE, ML_IMG_SIZE))
    img = img / 255.0
    img = img.flatten()
    return np.expand_dims(img, axis=0)  # (1, 32*32*3)

def preprocess_cnn(image):
    """For CNN"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (DL_IMG_SIZE, DL_IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)  # (1, 128, 128, 3)

def preprocess_vgg(image):
    """For VGG16"""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (DL_IMG_SIZE, DL_IMG_SIZE))
    img = preprocess_input(img.astype(np.float32))
    return np.expand_dims(img, axis=0)  # (1, 128, 128, 3)
# streamlit main UI

st.set_page_config(page_title="Garbage Classification Models", layout="wide")
st.title("🗑️ Garbage Classification Models Explorer")

# -----------------------------
# Sidebar UI
# -----------------------------
import streamlit as st

st.sidebar.title("About This App")
st.sidebar.info("""
Welcome to the **Garbage Image Classifier** App!  

**How to use:**  
1. **Upload an image** of an item you want to classify.  
2. **Choose a model** from the options:  
   - **Logistic Regression** (uses PCA for dimensionality reduction)  
   - **Random Forest** (works on raw image features)  
   - **CNN / VGG16** (deep learning models)
3. Click **Predict** to see the predicted class.  

The app will display the **predicted category** of your uploaded image.  
Easy, fast, and interactive!  
""")

st.sidebar.header("⚙️ Controls")
model_choice = st.sidebar.radio("Choose Model", ["Logistic Regression", "Random Forest", "CNN", "VGG16"])
live_train = st.sidebar.checkbox("Enable Live Training (Overfitting Demo)", value=False)

# -----------------------------
# Load the selected model
# -----------------------------
model, extra = load_selected_model(model_choice)

if model is not None:
    st.success(f"{model_choice} loaded successfully ✅")
else:
    st.error(f"Failed to load {model_choice}. Check logs!")

# -----------------------------
# Model Overview
# -----------------------------
st.subheader("📌 Model Overview")

if model_choice == "Logistic Regression":
    st.markdown("""
    **Logistic Regression (Baseline ML Model)**  
    - Uses flattened pixel values  
    - PCA used for dimensionality reduction
    """)

elif model_choice == "Random Forest":
    st.markdown("""
    **Random Forest (ML Model)**  
    - Uses flattened pixel values  
    - PCA used for dimensionality reduction
    """)

elif model_choice == "CNN":
    st.markdown("""
    **CNN (Deep Learning Model)**  
    - Convolutional layers extract features  
    - Softmax output over garbage classes
    """)

elif model_choice == "VGG16":
    st.markdown("""
    **VGG16 (Transfer Learning)**  
    - Pretrained VGG16 backbone  
    - Fine-tuned for garbage classification
    """)

# -----------------------------
# Image Upload and Prediction   

st.subheader("📷 Upload Image for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#-_-----

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", width=200)

    # -----------------------------
    # Prediction
    # -----------------------------
    if st.button("Predict"):
        # if model_choice in ["Logistic Regression", "Random Forest"]:
        #     preprocessed_img = preprocess_ml(image)
        #     pca = extra
        #     img_pca = pca.transform(preprocessed_img)
        #     prediction = model.predict(img_pca)
        #     predicted_class = class_names[prediction[0]]
        if model_choice == "Logistic Regression":
    # Preprocess image for Logistic Regression
            preprocessed_img = preprocess_ml(image)  # Flattened array
            # Apply PCA ( PCA was fitted on training data)
            pca = extra  # 'extra' should be your fitted PCA object
            img_pca = pca.transform(preprocessed_img.reshape(1, -1))  # 1 sample, reduced features
            prediction = model.predict(img_pca)
            predicted_class = class_names[prediction[0]]
            st.success(f"Logistic Regression Prediction: {predicted_class}")

        elif model_choice == "Random Forest":
            # Preprocess image for Random Forest
            preprocessed_img = preprocess_ml(image)  # Flattened array
            img_flat = preprocessed_img.reshape(1, -1)  # 1 sample, original features
            prediction = model.predict(img_flat)
            predicted_class = class_names[prediction[0]]
            st.success(f"Random Forest Prediction: {predicted_class}")

        elif model_choice == "CNN":
            preprocessed_img = preprocess_cnn(image)
            prediction = model.predict(preprocessed_img)
            predicted_class = class_names[np.argmax(prediction)]

        elif model_choice == "VGG16":
            preprocessed_img = preprocess_vgg(image)
            prediction = model.predict(preprocessed_img)
            predicted_class = class_names[np.argmax(prediction)]

        # Store in session_state
        st.session_state["predicted_class"] = predicted_class
        st.session_state["preprocessed_img"] = preprocessed_img

    

    # -----------------------------
    # Live Training (Overfitting Demo)
    # -----------------------------
    if live_train:

        if model_choice not in ["CNN", "VGG16"]:
            st.warning("Live training is only available for CNN and VGG16 models.")
        
        elif "predicted_class" not in st.session_state:
            st.info("Please run prediction first before live training.")

        else:
            st.subheader("🔥 Live Training (Overfitting Demo)")
            st.info(
                "This intentionally overfits the model on ONE image to demonstrate memorization."
            )

            epochs = st.slider("Select number of epochs", 1, 50, 10)

            if st.button("Start Live Training"):
                labels = np.zeros((1, NUM_CLASSES))
                labels[0, class_names.index(st.session_state["predicted_class"])] = 1
                
                
                from tensorflow.keras.optimizers import Adam

                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
                )

                history = model.fit(
                    st.session_state["preprocessed_img"],
                    labels,
                    epochs=epochs,
                    verbose=1
                )

                st.success("Live training completed!")

                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(1, 2, figsize=(12, 4))

                ax[0].plot(history.history["loss"])
                ax[0].set_title("Training Loss")
                ax[0].set_xlabel("Epoch")
                ax[0].set_ylabel("Loss")

                ax[1].plot(history.history["accuracy"])
                ax[1].set_title("Training Accuracy")
                ax[1].set_xlabel("Epoch")
                ax[1].set_ylabel("Accuracy")

                st.pyplot(fig)

# if uploaded_file is not None:
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#     st.image(image, caption='Uploaded Image', width=200)

#     if st.button("Predict"):
#         if model_choice in ["Logistic Regression", "Random Forest"]:
#             preprocessed_img = preprocess_ml(image)
#             pca = extra
#             img_pca = pca.transform(preprocessed_img)
#             prediction = model.predict(img_pca)
#             predicted_class = class_names[prediction[0]]
#             st.success(f"Predicted Class: {predicted_class}")
        
#         elif model_choice == "CNN":
#             preprocessed_img = preprocess_cnn(image)
#             prediction = model.predict(preprocessed_img)
#             predicted_class = class_names[np.argmax(prediction)]
#             st.success(f"Predicted Class: {predicted_class}")

#         elif model_choice == "VGG16":
#             preprocessed_img = preprocess_vgg(image)
#             prediction = model.predict(preprocessed_img)
#             predicted_class = class_names[np.argmax(prediction)]
#             st.success(f"Predicted Class: {predicted_class}")
#             # ----------------------------- 
#             # Live Training (Overfitting Demo)
#         if live_train:
            
#             st.subheader("🚀 Live Training (Overfitting Demo)")

#             epochs = st.slider("Select number of epochs", 1, 100, 10)

#             if st.button("Start Live Training", key="live_train_btn"):
#                 labels = np.zeros((1, NUM_CLASSES))
#                 labels[0, class_names.index(predicted_class)] = 1

#                 model.compile(
#                     optimizer="adam",
#                     loss="categorical_crossentropy",
#                     metrics=["accuracy"]
#                 )

#                 history = model.fit(
#                     preprocessed_img,
#                     labels,
#                     epochs=epochs,
#                     verbose=1
#                 )

#                 st.success("Live training completed!")

#                 fig, ax = plt.subplots(1, 2, figsize=(12, 4))
#                 ax[0].plot(history.history["loss"])
#                 ax[0].set_title("Training Loss")
#                 ax[0].set_xlabel("Epoch")
#                 ax[0].set_ylabel("Loss")
#                 ax[1].plot(history.history["accuracy"])
#                 ax[1].set_title("Training Accuracy")
#                 ax[1].set_xlabel("Epoch")
#                 ax[1].set_ylabel("Accuracy")
#                 st.pyplot(fig)
                
        
                        
                      
            else:
                 st.warning("Live training is only available for CNN and VGG16 models.")
        # ----------------------------- 
        #bar chart confidence scores
        if uploaded_file is not None and model is not None:
            st.subheader("📊 Prediction Confidence Scores")
            if model_choice in ["Logistic Regression", "Random Forest"]:
                preprocessed_img = preprocess_ml(image)
                pca = extra
                img_pca = pca.transform(preprocessed_img)
                probabilities = model.predict_proba(img_pca)[0]
            elif model_choice == "CNN":
                preprocessed_img = preprocess_cnn(image)
                probabilities = model.predict(preprocessed_img)[0]
            elif model_choice == "VGG16":
                preprocessed_img = preprocess_vgg(image)
                probabilities = model.predict(preprocessed_img)[0]

            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt

            df = pd.DataFrame({
                'Class': class_names,
                'Confidence': probabilities
            })

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Confidence', y='Class', data=df, ax=ax)
            ax.set_title('Prediction Confidence Scores')
            st.pyplot(fig)  
# -----------------------------
# Footer

st.markdown("---")
st.markdown("Developed by sonia firdous")