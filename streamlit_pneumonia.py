import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# from keras.preprocessing.image import ImageDataGenerator
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from textblob import TextBlob
import tweepy
import requests

# from mesa import Agent, Model
from mesa.agent import Agent
from mesa.model import Model

from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Fetch credentials securely
bearer_token = os.getenv("BEARER_tOKEN")
news_api_key = os.getenv("NEWS_API_KEY")


# -------------------- Session State Initialization --------------------
if "run_triggered" not in st.session_state:
    st.session_state.run_triggered = False



# -------------------- Streamlit UI --------------------
st.title("🧠 Pneumonia Misinformation Simulation")
st.markdown("A simulation of misinformation's effect on patient care-seeking behavior.")

# -------------------- Sidebar Configuration --------------------
st.sidebar.header("Configuration")

tweet_query = st.sidebar.text_input("Twitter Search Query", "Pneumonia")
tweet_count = st.sidebar.slider("Number of Tweets", 1, 100, 10)
train_dir = st.sidebar.text_input("Path to Train Images", "train/")
test_dir = st.sidebar.text_input("Path to Test Images", "test/")

run_button = st.sidebar.button("Run Simulation")
reset_button = st.sidebar.button("Reset Simulation")

if reset_button:
    st.session_state.run_triggered = False
    st.rerun()

if run_button:
    st.session_state.run_triggered = True


if st.session_state.run_triggered:

    # -------------------- 1. Data Preprocessing --------------------
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    try:
        train_data = train_datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='binary')
        test_data = test_datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=32, class_mode='binary')

        # X_train, y_train = train_data.next()
        X_train, y_train = next(train_data)

        X_train = X_train.reshape(X_train.shape[0], -1)

        # X_test, y_test = test_data.next()
        X_test, y_test = next(test_data)
        X_test = X_test.reshape(X_test.shape[0], -1)

        log_reg_model = LogisticRegression(max_iter=1000)
        log_reg_model.fit(X_train, y_train)
        log_reg_pred = log_reg_model.predict(X_test)

        xgb_model = XGBClassifier()
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        def evaluate_model(y_true, y_pred):
            return accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred), classification_report(y_true, y_pred)

        log_acc, log_cm, log_cr = evaluate_model(y_test, log_reg_pred)
        xgb_acc, xgb_cm, xgb_cr = evaluate_model(y_test, xgb_pred)

        st.subheader("🧪 Model Evaluation")
        st.markdown("**Logistic Regression**")
        st.text(f"Accuracy: {log_acc:.2f}")
        st.text(log_cr)

        st.markdown("**XGBoost**")
        st.text(f"Accuracy: {xgb_acc:.2f}")
        st.text(xgb_cr)

    except Exception as e:
        st.error(f"Error in image preprocessing or model training: {e}")

    # -------------------- 2. Twitter Data Collection --------------------
    tweets = []
    if bearer_token:
        try:
            client = tweepy.Client(bearer_token=bearer_token)
            results = client.search_recent_tweets(query=tweet_query, max_results=tweet_count, tweet_fields=["text"])
            tweets = [tweet.text for tweet in results.data] if results.data else []
            st.success(f"✅ Collected {len(tweets)} tweets.")
        except Exception as e:
            st.error(f"Error fetching tweets: {e}")
    else:
        st.warning("⚠️ Please provide a valid Twitter Bearer Token.")

    # -------------------- 3. News Data Collection --------------------
    news_articles = []
    if news_api_key:
        try:
            news_url = f'https://newsapi.org/v2/everything?q=pneumonia&apiKey={news_api_key}'
            response = requests.get(news_url)
            news_data = response.json()
            articles = news_data.get('articles', [])
            news_articles = [
                (a.get('title') or '') + " " + (a.get('description') or '')
                for a in articles if a.get('title') or a.get('description')
            ]
            st.success(f"✅ Collected {len(news_articles)} news articles.")
        except Exception as e:
            st.error(f"Error fetching news: {e}")
    else:
        st.warning("⚠️ Please provide a valid News API Key.")

    # -------------------- 4. Misinformation Detection --------------------
    def detect_misinformation(texts):
        return [1 if TextBlob(t).sentiment.polarity < 0 else 0 for t in texts]

    tweet_misinfo = detect_misinformation(tweets)
    news_misinfo = detect_misinformation(news_articles)
    avg_misinfo_score = (sum(tweet_misinfo) + sum(news_misinfo)) / (len(tweet_misinfo) + len(news_misinfo) + 1e-5)
    st.metric("💬 Misinformation Rate", f"{avg_misinfo_score:.2f}")

    # -------------------- 5. Agent-Based Model --------------------
    class Patient(Agent):
        def __init__(self, uid, model, misinfo_score=0.5):
            Agent.__init__(self, uid, model)   # <-- explicit call
            self.symptom_severity = random.choice([0, 1])
            self.trust_in_clinician = 0.5
            self.misinformation_exposure = misinfo_score
            self.symptom_reporting_accuracy = 1
            self.care_seeking_behavior = 0.5


        def step(self):
            if self.misinformation_exposure > 0.7 and random.random() < 0.4:
                self.symptom_severity = 0
            elif self.trust_in_clinician > 0.8:
                self.symptom_severity = 1

            if self.misinformation_exposure > 0.7:
                self.care_seeking_behavior = max(0, self.care_seeking_behavior - 0.3)
            elif self.symptom_severity == 1 and self.trust_in_clinician > 0.5:
                self.care_seeking_behavior = min(1, self.care_seeking_behavior + 0.5)

    class Clinician(Agent):
        def __init__(self, uid, model):
            Agent.__init__(self, uid, model)   # <-- explicit
            self.trust_in_patient = 0.5


        def step(self): pass

    class MisinformationModel(Model):
        def __init__(self, num_patients, width, height, misinfo_level=0.5):
            self.grid = MultiGrid(width, height, True)
            self.schedule = RandomActivation(self)
            for i in range(num_patients):
                mis_score = random.choice([misinfo_level, 0.8])
                patient = Patient(i, self, mis_score)
                self.schedule.add(patient)
                self.grid.place_agent(patient, (self.random.randrange(width), self.random.randrange(height)))

            clinician = Clinician(num_patients, self)
            self.schedule.add(clinician)
            self.grid.place_agent(clinician, (self.random.randrange(width), self.random.randrange(height)))

            self.datacollector = DataCollector(
                agent_reporters={
                    "Symptom Severity": "symptom_severity",
                    "Care Seeking Behavior": "care_seeking_behavior"
                }
            )

        def step(self):
            self.datacollector.collect(self)
            self.schedule.step()

    # -------------------- 6. Run Simulation --------------------
    sim_model = MisinformationModel(num_patients=10, width=10, height=10, misinfo_level=avg_misinfo_score)
    for _ in range(100):
        sim_model.step()

    df = sim_model.datacollector.get_agent_vars_dataframe()

    # -------------------- 7. Visualization --------------------
    st.subheader("📉 Simulation Results")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=df['Symptom Severity'], y=df['Care Seeking Behavior'], alpha=0.7, ax=ax1)
    ax1.set_title('Impact of Misinformation on Care-Seeking')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(
        x=df['Symptom Severity'],
        y=df['Care Seeking Behavior'],
        hue=df['Symptom Severity'],
        size=df['Care Seeking Behavior'],
        palette="coolwarm",
        ax=ax2
    )
    ax2.set_title('Symptom Severity vs Care-Seeking')
    st.pyplot(fig2)
