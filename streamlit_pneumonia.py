# pneumonia_v07.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
import random
import requests
import os
import re
from PIL import Image
from urllib.parse import quote_plus
import pickle
from datetime import datetime

# Resolve default dataset directories relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRAIN_DIR = os.path.join(SCRIPT_DIR, "train")
DEFAULT_VAL_DIR = os.path.join(SCRIPT_DIR, "val")
DEFAULT_TEST_DIR = os.path.join(SCRIPT_DIR, "test")
MODELS_DIR = os.path.join(SCRIPT_DIR, "saved_models")

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from textblob import TextBlob

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from sklearn.feature_extraction.text import TfidfVectorizer


# =======================
# 1) MODEL TRAINING (from v06, with persistence via session_state)
# =======================

@st.cache_resource
def train_models(train_dir, valid_dir, test_dir, img_size=(64, 64), batch_size=32):
    # ImageDataGenerators for train and validation
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size,
        class_mode='binary', shuffle=True
    )
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir, target_size=img_size, batch_size=batch_size,
        class_mode='binary', shuffle=False
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir, target_size=img_size, batch_size=batch_size,
        class_mode='binary', shuffle=False
    )

    # Prepare data arrays (small datasets assumed)
    def generator_to_xy(generator):
        X, y = [], []
        for _ in range(len(generator)):
            x_batch, y_batch = next(generator)
            X.append(x_batch)
            y.append(y_batch)
        return np.vstack(X), np.hstack(y)

    X_train, y_train = generator_to_xy(train_generator)
    X_valid, y_valid = generator_to_xy(valid_generator)
    X_test, y_test = generator_to_xy(test_generator)

    # Flatten images for Logistic Regression & XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_valid_flat = X_valid.reshape(X_valid.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Train Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_flat, y_train)
    y_pred_log_val = log_reg.predict(X_valid_flat)
    acc_log = accuracy_score(y_valid, y_pred_log_val)
    cm_log = confusion_matrix(y_valid, y_pred_log_val)
    cr_log = classification_report(y_valid, y_pred_log_val, output_dict=False)
    # Test metrics
    y_pred_log_test = log_reg.predict(X_test_flat)
    acc_log_test = accuracy_score(y_test, y_pred_log_test)
    cm_log_test = confusion_matrix(y_test, y_pred_log_test)
    cr_log_test = classification_report(y_test, y_pred_log_test, output_dict=False)

    # Train XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_flat, y_train)
    y_pred_xgb_val = xgb.predict(X_valid_flat)
    acc_xgb = accuracy_score(y_valid, y_pred_xgb_val)
    cm_xgb = confusion_matrix(y_valid, y_pred_xgb_val)
    cr_xgb = classification_report(y_valid, y_pred_xgb_val, output_dict=False)
    # Test metrics
    y_pred_xgb_test = xgb.predict(X_test_flat)
    acc_xgb_test = accuracy_score(y_test, y_pred_xgb_test)
    cm_xgb_test = confusion_matrix(y_test, y_pred_xgb_test)
    cr_xgb_test = classification_report(y_test, y_pred_xgb_test, output_dict=False)

    return {
        "log_reg": log_reg,
        "xgb": xgb,
        "val_data": (X_valid_flat, y_valid),
        "test_data": (X_test_flat, y_test),
        "metrics": {
            "log_reg": {
                "accuracy": acc_log,
                "conf_matrix": cm_log,
                "class_report": cr_log,
                "test_accuracy": acc_log_test,
                "test_conf_matrix": cm_log_test,
                "test_class_report": cr_log_test
            },
            "xgb": {
                "accuracy": acc_xgb,
                "conf_matrix": cm_xgb,
                "class_report": cr_xgb,
                "test_accuracy": acc_xgb_test,
                "test_conf_matrix": cm_xgb_test,
                "test_class_report": cr_xgb_test
            }
        }
    }

# =======================
# 2) IMAGE PREPROCESSING (from v06)
# =======================

def preprocess_image(uploaded_file, img_size=(64, 64)):
    img = Image.open(uploaded_file).convert('RGB').resize(img_size)
    img_array = image.img_to_array(img) / 255.0
    return img_array.reshape(1, -1)

# =======================
# 3) MISINFORMATION DETECTION & DATA (from v06)
# =======================

def detect_misinformation(texts):
    results = []
    for text in texts:
        polarity = TextBlob(text).sentiment.polarity
        tag = "❌ Misinformation" if polarity < 0 else "✅ Trusted"
        results.append((text, tag))
    return results

def raphael_score_claim(claim_text):
    pneumonia_keywords = ["pneumonia", "lung infection", "respiratory"]
    harmful = any(word in claim_text.lower() for word in pneumonia_keywords)
    return {
        "claim": claim_text,
        "checkworthy": True,
        "harmful": harmful,
        "needs_citation": True,
        "confidence": 0.85 if harmful else 0.5
    }

def get_reddit_posts(query='pneumonia', size=50):
    """Get Reddit posts using Reddit's search API (free, no auth required)"""
    try:
        reddit_url = f"https://www.reddit.com/search.json?q={quote_plus(query)}&limit={size}&sort=new"
        headers = {"User-Agent": "Mozilla/5.0 (StreamlitApp)"}
        response = requests.get(reddit_url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            children = data.get("data", {}).get("children", [])
            texts = []
            for child in children:
                title = child.get("data", {}).get("title", "") or ""
                selftext = child.get("data", {}).get("selftext", "") or ""
                text = f"{title} {selftext}".strip()
                if text:
                    texts.append(text)
            return texts
        else:
            st.warning(f"⚠️ Reddit search returned status {response.status_code}.")
            return []
    except Exception as e:
        st.error(f"Error fetching Reddit data: {e}")
        return []

def get_tavily_results(query='pneumonia', size=20, api_key=None):
    """Get web search results using Tavily API"""
    if not api_key:
        return []
    
    try:
        tavily_payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": size,
            "include_raw_content": True,
        }
        response = requests.post("https://api.tavily.com/search", json=tavily_payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            texts = []
            for result in results:
                content = result.get("content") or result.get("raw_content") or ""
                if content:
                    texts.append(content)
            return texts
        else:
            st.warning(f"⚠️ Tavily search returned status {response.status_code}.")
            return []
    except Exception as e:
        st.error(f"Error fetching Tavily results: {e}")
        return []

def get_wikipedia_results(query='pneumonia', size=20):
    """Get Wikipedia search results (free, no auth required)"""
    try:
        wiki_url = f"https://en.wikipedia.org/w/rest.php/v1/search/page?q={quote_plus(query)}&limit={size}"
        response = requests.get(wiki_url, timeout=20)
        if response.status_code == 200:
            data = response.json()
            pages = data.get("pages", [])
            texts = []
            for page in pages:
                title = page.get("title") or ""
                excerpt = page.get("excerpt") or ""
                # Strip HTML tags in excerpt
                excerpt_clean = re.sub(r"<[^>]+>", " ", excerpt)
                text = f"{title} {excerpt_clean}".strip()
                if text:
                    texts.append(text)
            return texts
        else:
            st.warning(f"⚠️ Wikipedia search returned status {response.status_code}.")
            return []
    except Exception as e:
        st.error(f"Error fetching Wikipedia results: {e}")
        return []

def get_hackernews_results(query='pneumonia', size=20):
    """Get Hacker News search results (free via Algolia API)"""
    try:
        hn_url = f"https://hn.algolia.com/api/v1/search?query={quote_plus(query)}&tags=story&hitsPerPage={size}"
        response = requests.get(hn_url, timeout=20)
        if response.status_code == 200:
            data = response.json()
            hits = data.get("hits", [])
            texts = []
            for hit in hits:
                title = hit.get("title") or ""
                story_text = hit.get("story_text") or hit.get("_highlightResult", {}).get("title", {}).get("value", "") or ""
                story_text_clean = re.sub(r"<[^>]+>", " ", str(story_text))
                text = f"{title} {story_text_clean}".strip()
                if text:
                    texts.append(text)
            return texts
        else:
            st.warning(f"⚠️ Hacker News search returned status {response.status_code}.")
            return []
    except Exception as e:
        st.error(f"Error fetching Hacker News results: {e}")
        return []

def clean_text_for_analysis(text):
    """Clean text for better sentiment analysis"""
    if not text:
        return ""
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', str(text).strip())
    # Remove very short texts that might skew analysis
    if len(text) < 10:
        return ""
    return text

def get_data_source_info(source):
    """Get information about data sources"""
    info = {
        "Reddit (Free API)": "Real-time Reddit posts and discussions",
        "Tavily Web Search": "Comprehensive web search results",
        "Wikipedia (Free)": "Academic and factual information",
        "Hacker News (Free)": "Tech community discussions and news",
        "HealthVer (local CSV)": "Health verification CSVs in local 'data' folder",
        "HealthVer (local JSON)": "Health verification dataset",
        "FullFact (local JSON)": "Fact-checking dataset"
    }
    return info.get(source, "Unknown source")

# =======================
# 4) AGENT-BASED SIMULATION (v07: adds Clinician + slider-controlled misinfo exposure)
# =======================

class Patient(Agent):
    def __init__(self, unique_id, model, misinformation_score=0.5):
        super().__init__(unique_id, model)
        self.symptom_severity = random.choice([0, 1])
        self.trust_in_clinician = 0.5
        self.misinformation_exposure = misinformation_score
        self.care_seeking_behavior = 0.5

    def step(self):
        # Misinformation reduces symptom perception and care seeking
        if self.misinformation_exposure > 0.7 and random.random() < 0.4:
            self.symptom_severity = 0
        # Trust increases symptom recognition
        elif self.trust_in_clinician > 0.8:
            self.symptom_severity = 1

        # Care seeking behavior adjusted by misinformation and trust
        if self.misinformation_exposure > 0.7:
            self.care_seeking_behavior = max(0, self.care_seeking_behavior - 0.3)
        elif self.symptom_severity == 1 and self.trust_in_clinician > 0.5:
            self.care_seeking_behavior = min(1, self.care_seeking_behavior + 0.5)

class Clinician(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        # Find patients in the same cell
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        patients_here = [agent for agent in cellmates if isinstance(agent, Patient)]
        for patient in patients_here:
            # Increase patient trust if clinician present
            patient.trust_in_clinician = min(1.0, patient.trust_in_clinician + 0.1)
            # Potentially decrease misinformation exposure
            if patient.misinformation_exposure > 0:
                patient.misinformation_exposure = max(0, patient.misinformation_exposure - 0.05)

class MisinformationModel(Model):
    def __init__(self, num_patients, num_clinicians, width, height, misinformation_exposure):
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            agent_reporters={
                "Symptom Severity": "symptom_severity",
                "Care Seeking Behavior": "care_seeking_behavior",
                "Trust in Clinician": "trust_in_clinician",
                "Misinformation Exposure": "misinformation_exposure"
            }
        )

        # Add patients
        for i in range(num_patients):
            patient = Patient(i, self, misinformation_exposure)
            self.schedule.add(patient)
            x, y = self.random.randrange(width), self.random.randrange(height)
            self.grid.place_agent(patient, (x, y))

        # Add clinicians
        for i in range(num_patients, num_patients + num_clinicians):
            clinician = Clinician(i, self)
            self.schedule.add(clinician)
            x, y = self.random.randrange(width), self.random.randrange(height)
            self.grid.place_agent(clinician, (x, y))

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# =======================
# 5) STREAMLIT UI
# =======================

st.set_page_config(page_title="🩺 Pneumonia & Misinformation Simulator", layout="wide")
st.title("🩺 Pneumonia Diagnosis & Misinformation Simulator")

# Add dashboard overview
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
    <h3 style="color: white; margin-top: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">📊 Dashboard Overview</h3>
    <p style="color: white; opacity: 0.95;">This comprehensive tool combines:</p>
    <ul style="color: white; opacity: 0.9;">
        <li><strong>🔬 AI-Powered X-ray Analysis:</strong> Advanced pneumonia detection using ML models</li>
        <li><strong>🌐 Multi-Source Data Collection:</strong> Real-time analysis from Reddit, Wikipedia, Hacker News, and more</li>
        <li><strong>📈 Advanced Analytics:</strong> Sentiment analysis, misinformation detection, and interactive visualizations</li>
        <li><strong>🎯 Agent-Based Simulation:</strong> Model the impact of misinformation on healthcare behavior</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar — dataset & training controls (from v06)
st.sidebar.header("Dataset & Model Training")
train_dir = st.sidebar.text_input(
    "Training Images Folder (local path)",
    DEFAULT_TRAIN_DIR,
    help="Folder containing 'NORMAL' and 'PNEUMONIA' subfolders (e.g., train)"
)
valid_dir = st.sidebar.text_input(
    "Validation Images Folder (local path)",
    DEFAULT_VAL_DIR,
    help="Folder containing 'NORMAL' and 'PNEUMONIA' subfolders (e.g., val)"
)
test_dir = st.sidebar.text_input(
    "Test Images Folder (local path)",
    DEFAULT_TEST_DIR,
    help="Folder containing 'NORMAL' and 'PNEUMONIA' subfolders (e.g., test)"
)
train_button = st.sidebar.button("Train Models on Dataset")

# API Keys (optional)
tavily_api_key = st.sidebar.text_input("Tavily API Key (optional)", type="password", help="Get free API key from tavily.com")

# Data source selection
dataset_source = st.sidebar.selectbox(
    "Misinformation Source Dataset",
    ["Reddit (Free API)", "Tavily Web Search", "Wikipedia (Free)", "Hacker News (Free)", "HealthVer (local CSV)", "HealthVer (local JSON)", "FullFact (local JSON)"]
)

# Search configuration
search_query = st.sidebar.text_input("Search Keyword", value="pneumonia")
if dataset_source in ["Reddit (Free API)", "Tavily Web Search", "Wikipedia (Free)", "Hacker News (Free)"]:
    search_count = st.sidebar.slider("Number of Results", 5, 50, 20)

# Show data source information
if dataset_source:
    st.sidebar.info(f"📚 **{dataset_source}**: {get_data_source_info(dataset_source)}")

# Add sidebar status indicators
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Session Status")

# Initialize session state for tracking
if 'data_collected' not in st.session_state:
    st.session_state.data_collected = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False

# Status indicators
status_col1, status_col2 = st.sidebar.columns(2)
with status_col1:
    model_status = "✅" if "model_data" in st.session_state else "⏳"
    st.write(f"{model_status} Models")
with status_col2:
    data_status = "✅" if st.session_state.data_collected else "⏳"
    st.write(f"{data_status} Data")

model_choice = st.sidebar.radio("Choose X-ray Model for Prediction", ("Logistic Regression", "XGBoost"))
uploaded_file = st.sidebar.file_uploader("Upload Chest X-Ray Image", type=["jpg", "jpeg", "png"])

# Agent-Based Simulation Controls (v07 additions)
st.subheader("3⃣ Agent-Based Misinformation Simulation")
num_agents = st.sidebar.slider("Number of Patient Agents", 5, 50, 10)
num_clinicians = st.sidebar.slider("Number of Clinician Agents", 1, 10, 3)
misinfo_exposure = st.sidebar.slider("Baseline Misinformation Exposure", 0.0, 1.0, 0.5, 0.05)
simulate_button = st.sidebar.button("Run Agent-Based Simulation")

# ===============================
# 6. HealthVer Dataset Evaluation
# ===============================
st.markdown("## 📊 HealthVer Benchmark Evaluation")

@st.cache_data
def load_healthver_data():
    # train_df = pd.read_csv("data/healthver_train.csv", sep="\t")
    # dev_df = pd.read_csv("data/healthver_dev.csv", sep="\t")
    # test_df = pd.read_csv("data/healthver_test.csv", sep="\t")
    
    train_df = pd.read_csv("data/healthver_train.csv", sep=None, engine="python")
    dev_df = pd.read_csv("data/healthver_dev.csv", sep=None, engine="python")
    test_df = pd.read_csv("data/healthver_test.csv", sep=None, engine="python")
    return train_df, dev_df, test_df

try:
    train_df, dev_df, test_df = load_healthver_data()

    # Encode labels
    label_map = {"Supports": 0, "Refutes": 1, "Neutral": 2}
    train_df["label_enc"] = train_df["label"].map(label_map)
    dev_df["label_enc"] = dev_df["label"].map(label_map)
    test_df["label_enc"] = test_df["label"].map(label_map)

    # Feature: evidence + claim concatenation
    def combine_text(df):
        return (df["evidence"].fillna("") + " " + df["claim"].fillna("")).values

    X_train_text = combine_text(train_df)
    y_train = train_df["label_enc"].values
    X_dev_text = combine_text(dev_df)
    y_dev = dev_df["label_enc"].values
    X_test_text = combine_text(test_df)
    y_test = test_df["label_enc"].values

    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    X_dev = vectorizer.transform(X_dev_text)
    X_test = vectorizer.transform(X_test_text)

    # Train Logistic Regression (baseline)
    clf = LogisticRegression(max_iter=300)
    clf.fit(X_train, y_train)

    # Evaluate
    y_dev_pred = clf.predict(X_dev)
    y_test_pred = clf.predict(X_test)

    dev_acc = accuracy_score(y_dev, y_dev_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    st.markdown(f"**✅ Dev Accuracy:** {dev_acc:.3f}")
    st.markdown(f"**✅ Test Accuracy:** {test_acc:.3f}")

    # Classification Report
    st.text("Classification Report (Test Set):")
    st.text(classification_report(y_test, y_test_pred, target_names=label_map.keys()))

except Exception as e:
    st.error(f"⚠️ Could not evaluate HealthVer dataset: {e}")


# =======================
# TRAIN MODELS (from v06; now persisted)
# =======================

if train_button:
    if (
        train_dir and valid_dir and test_dir and
        os.path.exists(train_dir) and os.path.exists(valid_dir) and os.path.exists(test_dir)
    ):
        with st.spinner("Training models on your dataset..."):
            model_data = train_models(train_dir, valid_dir, test_dir)
            st.session_state["model_data"] = model_data
            st.session_state.model_trained = True
        st.success("Models trained!")

        # Save trained models
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(MODELS_DIR, f"log_reg_{timestamp}.pkl")
            xgb_path = os.path.join(MODELS_DIR, f"xgb_{timestamp}.pkl")
            with open(log_path, "wb") as f:
                pickle.dump(model_data['log_reg'], f)
            with open(xgb_path, "wb") as f:
                pickle.dump(model_data['xgb'], f)
            st.info(f"💾 Saved models to:\n- {log_path}\n- {xgb_path}")
        except Exception as e:
            st.warning(f"Could not save models: {e}")

        # Show evaluation results
        st.write("### Validation Accuracy:")
        st.write(f"Logistic Regression: {model_data['metrics']['log_reg']['accuracy']:.2f}")
        st.write(f"XGBoost: {model_data['metrics']['xgb']['accuracy']:.2f}")

        st.write("### Test Accuracy:")
        st.write(f"Logistic Regression: {model_data['metrics']['log_reg']['test_accuracy']:.2f}")
        st.write(f"XGBoost: {model_data['metrics']['xgb']['test_accuracy']:.2f}")

        st.write("### Classification Reports:")
        st.text("Logistic Regression:")
        st.text(model_data['metrics']['log_reg']['class_report'])
        st.text("XGBoost:")
        st.text(model_data['metrics']['xgb']['class_report'])

        st.write("### Test Classification Reports:")
        st.text("Logistic Regression (Test):")
        st.text(model_data['metrics']['log_reg']['test_class_report'])
        st.text("XGBoost (Test):")
        st.text(model_data['metrics']['xgb']['test_class_report'])

        # Confusion matrices side-by-side
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(model_data['metrics']['log_reg']['conf_matrix'], annot=True, fmt='d', ax=axs[0])
        axs[0].set_title("Logistic Regression Confusion Matrix")
        sns.heatmap(model_data['metrics']['xgb']['conf_matrix'], annot=True, fmt='d', ax=axs[1])
        axs[1].set_title("XGBoost Confusion Matrix")
        st.pyplot(fig)

        # Test confusion matrices side-by-side
        fig_test, axs_test = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(model_data['metrics']['log_reg']['test_conf_matrix'], annot=True, fmt='d', ax=axs_test[0])
        axs_test[0].set_title("Logistic Regression Test Confusion Matrix")
        sns.heatmap(model_data['metrics']['xgb']['test_conf_matrix'], annot=True, fmt='d', ax=axs_test[1])
        axs_test[1].set_title("XGBoost Test Confusion Matrix")
        st.pyplot(fig_test)
    else:
        st.error("Please provide valid local paths for training, validation, and test image folders.")

# =======================
# X-RAY CLASSIFICATION (from v06; uses session_state for model persistence)
# =======================

st.subheader("1⃣ Chest X-Ray Pneumonia Classification")
if uploaded_file is not None:
    img_array = preprocess_image(uploaded_file)
    st.image(uploaded_file, caption="Uploaded Chest X-Ray", width=300)

    if "model_data" in st.session_state:
        model_data = st.session_state["model_data"]
        if model_choice == "Logistic Regression":
            pred = model_data['log_reg'].predict(img_array)[0]
        else:
            pred = model_data['xgb'].predict(img_array)[0]
        label = "Pneumonia" if pred == 1 else "Normal"
        st.success(f"Prediction: {label}")
    else:
        st.warning("Please train the models first to predict on uploaded images.")

# =======================
# MISINFORMATION TEXT ANALYSIS (from v06)
# =======================

st.subheader("2⃣ Misinformation Text Analysis")

texts = []
if dataset_source == "Reddit (Free API)":
    with st.spinner("Fetching Reddit posts..."):
        texts = get_reddit_posts(search_query, size=search_count)
    if texts:
        st.success(f"✅ Collected {len(texts)} Reddit posts.")
        st.session_state.data_collected = True

elif dataset_source == "Tavily Web Search":
    if tavily_api_key:
        with st.spinner("Searching web with Tavily..."):
            texts = get_tavily_results(search_query, size=search_count, api_key=tavily_api_key)
        if texts:
            st.success(f"✅ Collected {len(texts)} web results.")
            st.session_state.data_collected = True
    else:
        st.warning("⚠️ Please provide a Tavily API key to enable web search.")
        st.info("💡 Get a free API key from [tavily.com](https://tavily.com)")

elif dataset_source == "Wikipedia (Free)":
    with st.spinner("Searching Wikipedia..."):
        texts = get_wikipedia_results(search_query, size=search_count)
    if texts:
        st.success(f"✅ Collected {len(texts)} Wikipedia results.")
        st.session_state.data_collected = True

elif dataset_source == "Hacker News (Free)":
    with st.spinner("Searching Hacker News..."):
        texts = get_hackernews_results(search_query, size=search_count)
    if texts:
        st.success(f"✅ Collected {len(texts)} Hacker News stories.")
        st.session_state.data_collected = True

elif dataset_source == "HealthVer (local CSV)":
    # Controls for local CSV usage
    hv_split = st.sidebar.selectbox("HealthVer split (data folder)", ["train", "dev", "test"], index=1)
    hv_columns_selected = st.sidebar.multiselect(
        "Columns to analyze",
        ["claim", "evidence", "question"],
        default=["claim"]
    )
    csv_paths = {
        "train": os.path.join("data", "healthver_train.csv"),
        "dev": os.path.join("data", "healthver_dev.csv"),
        "test": os.path.join("data", "healthver_test.csv"),
    }
    csv_path = csv_paths.get(hv_split)
    if csv_path and os.path.exists(csv_path):
        try:
            df_hv = pd.read_csv(csv_path)
            use_cols = [c for c in hv_columns_selected if c in df_hv.columns]
            if not use_cols:
                # Fallback to any available known columns
                use_cols = [c for c in ["claim", "evidence", "question"] if c in df_hv.columns]
            if use_cols:
                # Concatenate selected columns' text
                texts = []
                for c in use_cols:
                    series_text = df_hv[c].dropna().astype(str).tolist()
                    texts.extend(series_text)
                if texts:
                    st.success(f"✅ Loaded {len(texts)} texts from {hv_split} CSV ({', '.join(use_cols)}).")
                    st.session_state.data_collected = True
                else:
                    st.warning("CSV loaded but no text found in selected columns.")
            else:
                st.warning("Selected columns not found in CSV. Available columns: " + ", ".join(df_hv.columns.astype(str)))
        except Exception as e:
            st.error(f"Failed to read HealthVer CSV: {e}")
    else:
        st.error(f"CSV not found at {csv_path}. Ensure the file exists in the 'data' folder.")

elif dataset_source == "HealthVer (local JSON)":
    healthver_file = st.sidebar.file_uploader("Upload HealthVer JSON dataset", type=["json"])
    if healthver_file:
        try:
            df_healthver = pd.read_json(healthver_file)
            texts = df_healthver['text'].tolist() if 'text' in df_healthver.columns else []
        except Exception as e:
            st.error(f"Failed to read HealthVer JSON: {e}")

elif dataset_source == "FullFact (local JSON)":
    fullfact_file = st.sidebar.file_uploader("Upload FullFact JSON dataset", type=["json"])
    if fullfact_file:
        try:
            df_fullfact = pd.read_json(fullfact_file)
            texts = df_fullfact['claim'].tolist() if 'claim' in df_fullfact.columns else []
        except Exception as e:
            st.error(f"Failed to read FullFact JSON: {e}")

if texts:
    if dataset_source == "Reddit (Free API)":
        st.markdown(f"### Latest Reddit posts mentioning '{search_query}'")
    elif dataset_source == "Tavily Web Search":
        st.markdown(f"### Web search results for '{search_query}'")
    elif dataset_source == "Wikipedia (Free)":
        st.markdown(f"### Wikipedia results for '{search_query}'")
    elif dataset_source == "Hacker News (Free)":
        st.markdown(f"### Hacker News stories about '{search_query}'")
    else:
        st.markdown("### Dataset posts")

    for post in texts[:5]:
        st.write(f"- {post[:200]}...")

    misinformation_results = detect_misinformation(texts[:10])
    st.markdown("### Misinformation Detection")
    for text, tag in misinformation_results:
        st.write(f"{tag}: {text[:150]}...")

    st.markdown("### RAPHAEL-style Claim Scoring")
    for post in texts[:5]:
        score = raphael_score_claim(post)
        st.write(
            f"Claim: {score['claim'][:100]}... | "
            f"Harmful: {score['harmful']} | "
            f"Confidence: {score['confidence']}"
        )
    
    # Additional analysis: Misinformation rate and sentiment analysis
    if texts:
        st.markdown("### 📊 Misinformation Analysis")
        
        # Clean texts for better analysis first
        try:
            cleaned_texts = [clean_text_for_analysis(text) for text in texts]
            cleaned_texts = [text for text in cleaned_texts if text]  # Remove empty texts
        except Exception as e:
            st.error(f"Error during text cleaning: {e}")
            cleaned_texts = texts  # Fallback to original texts
        
        # Data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 Total Texts", len(texts))
        with col2:
            avg_length = np.mean([len(text) for text in texts]) if texts else 0
            st.metric("📏 Avg Text Length", f"{avg_length:.0f} chars")
        with col3:
            # Calculate misinformation rate using cleaned texts
            if cleaned_texts:
                misinformation_flags = [1 if TextBlob(text).sentiment.polarity < 0 else 0 for text in cleaned_texts]
                misinfo_rate = sum(misinformation_flags) / len(misinformation_flags) if misinformation_flags else 0
                st.metric("💬 Misinformation Rate", f"{misinfo_rate:.2f}")
            else:
                st.metric("💬 Misinformation Rate", "N/A")
        
        # Show cleaning results
        if len(cleaned_texts) != len(texts):
            st.info(f"ℹ️ Text cleaning: {len(texts)} → {len(cleaned_texts)} valid texts")
        
        if not cleaned_texts:
            st.warning("⚠️ No valid texts found after cleaning for analysis.")
        else:
            # Sentiment distribution
            sentiment_scores = [TextBlob(text).sentiment.polarity for text in cleaned_texts]
            
            # Sentiment statistics
            st.markdown("### 📈 Sentiment Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("😊 Positive", f"{sum(1 for s in sentiment_scores if s > 0)}")
            with col2:
                st.metric("😐 Neutral", f"{sum(1 for s in sentiment_scores if s == 0)}")
            with col3:
                st.metric("😞 Negative", f"{sum(1 for s in sentiment_scores if s < 0)}")
            with col4:
                st.metric("📊 Mean", f"{np.mean(sentiment_scores):.3f}")
            
            # Show sample texts with their sentiment scores
            st.markdown("### 📝 Sample Texts with Sentiment Scores")
            sample_data = list(zip(cleaned_texts[:5], sentiment_scores[:5]))
            for text, sentiment in sample_data:
                sentiment_label = "❌ Negative" if sentiment < 0 else "✅ Positive" if sentiment > 0 else "⚪ Neutral"
                st.write(f"{sentiment_label} ({sentiment:.2f}): {text[:150]}...")
            


else:
    st.info("No text data loaded from selected dataset.")

# =======================
# AGENT-BASED SIMULATION (v07 with clinicians & exposure slider)
# =======================

if simulate_button:
    st.session_state.simulation_run = True
    model = MisinformationModel(num_agents, num_clinicians, 10, 10, misinfo_exposure)
    for _ in range(30):
        model.step()

    df_sim = model.datacollector.get_agent_vars_dataframe()
    st.write("### 📈 Simulation Results & Analysis")

    # Reset index for easier plotting
    df_reset = df_sim.reset_index()
    
    # Create multiple visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Original scatter plot with enhancements
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=df_reset,
            x="Symptom Severity",
            y="Care Seeking Behavior",
            hue="Trust in Clinician",
            size="Misinformation Exposure",
            alpha=0.7,
            ax=ax1,
            palette="coolwarm",
            sizes=(20, 200)
        )
        ax1.set_title("Impact of Misinformation & Trust on Care-Seeking")
        ax1.set_xlabel("Symptom Severity")
        ax1.set_ylabel("Care Seeking Behavior")
        st.pyplot(fig1)
    
    
    # 3. 2D Scatter Plot (converted from 3D)
    if len(df_reset) > 10:
        st.markdown("### 🎯 2D Relationship Analysis")
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(15, 6))
        
        # First 2D plot: Symptom Severity vs Care Seeking Behavior
        scatter1 = ax3a.scatter(df_reset['Symptom Severity'], 
                               df_reset['Care Seeking Behavior'],
                               c=df_reset['Misinformation Exposure'],
                               cmap='viridis', alpha=0.6, s=50)
        ax3a.set_xlabel('Symptom Severity')
        ax3a.set_ylabel('Care Seeking Behavior')
        ax3a.set_title('Symptoms vs Care-Seeking\n(Color = Misinformation Level)')
        plt.colorbar(scatter1, ax=ax3a, label='Misinformation Exposure', shrink=0.8)
        
        # Second 2D plot: Trust vs Care Seeking Behavior
        scatter2 = ax3b.scatter(df_reset['Trust in Clinician'], 
                               df_reset['Care Seeking Behavior'],
                               c=df_reset['Misinformation Exposure'],
                               cmap='viridis', alpha=0.6, s=50)
        ax3b.set_xlabel('Trust in Clinician')
        ax3b.set_ylabel('Care Seeking Behavior')
        ax3b.set_title('Trust vs Care-Seeking\n(Color = Misinformation Level)')
        plt.colorbar(scatter2, ax=ax3b, label='Misinformation Exposure', shrink=0.8)
        
        plt.tight_layout()
        st.pyplot(fig3)
    
    # 4. Summary statistics table
    st.markdown("### 📋 Simulation Summary Statistics")
    summary_stats = df_reset[["Symptom Severity", "Care Seeking Behavior", 
                             "Trust in Clinician", "Misinformation Exposure"]].describe()
    st.dataframe(summary_stats.round(3))

# =======================
# FOOTER
# =======================

st.markdown("---")
st.markdown(
    """
    💡 This app integrates:
    - Real Chest X-ray pneumonia classification with Logistic Regression and XGBoost on your datasets
    - Multi-source misinformation detection: Reddit (free API), Tavily web search, Wikipedia, Hacker News, HealthVer, FullFact
    - RAPHAEL-style claim scoring for health claims with sentiment analysis
    - Agent-based simulation modeling misinformation's impact on care-seeking behavior, with clinician interaction
    - Advanced visualizations: sentiment distributions, misinformation rates, and simulation trends
    """
)
