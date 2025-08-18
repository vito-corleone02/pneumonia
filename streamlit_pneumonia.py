import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import re

from keras.preprocessing.image import ImageDataGenerator
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from textblob import TextBlob
import tweepy
import requests
from urllib.parse import quote_plus

from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Fetch credentials securely (support legacy/mis-cased keys as fallback)
bearer_token = os.getenv("BEARER_TOKEN") 
news_api_key = os.getenv("NEWS_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")


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
reddit_count = st.sidebar.slider("Number of Reddit Posts", 1, 100, 10)
tavily_count = st.sidebar.slider("Number of Web Results (Tavily)", 1, 50, 10)
wiki_count = st.sidebar.slider("Number of Wikipedia Results", 1, 50, 10)
hn_count = st.sidebar.slider("Number of Hacker News Results", 1, 50, 10)
pubmed_count = st.sidebar.slider("Number of PubMed Results", 1, 50, 10)
crossref_count = st.sidebar.slider("Number of CrossRef Results", 1, 50, 10)
train_dir = st.sidebar.text_input("Path to Train Images", "train/")
test_dir = st.sidebar.text_input("Path to Test Images", "test/")
print_to_cli = st.sidebar.checkbox("Print search results to CLI", value=False)

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
        train_data = train_datagen.flow_from_directory(
            train_dir,
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary'
        )
        test_data = test_datagen.flow_from_directory(
            test_dir,
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary',
            shuffle=False  # deterministic evaluation on full test set
        )

        # Collect the full epoch (entire folder) for train and test
        def collect_full_epoch(data_iter):
            total = data_iter.samples
            batch_size = data_iter.batch_size
            steps = int(np.ceil(total / batch_size))
            feature_batches = []
            label_batches = []
            for _ in range(steps):
                xb, yb = data_iter.next()
                feature_batches.append(xb.reshape(xb.shape[0], -1))
                label_batches.append(np.asarray(yb).astype(int).ravel())
            X = np.vstack(feature_batches)
            y = np.concatenate(label_batches)
            # Guard against any over-collection due to iterator behavior
            return X[:total], y[:total]

        X_train, y_train = collect_full_epoch(train_data)
        X_test, y_test = collect_full_epoch(test_data)

        log_reg_model = LogisticRegression(max_iter=1000)
        log_reg_model.fit(X_train, y_train)
        log_reg_pred = log_reg_model.predict(X_test)

        xgb_model = XGBClassifier(eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        def evaluate_model(y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            cr_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            cr_df = pd.DataFrame(cr_dict).T
            # Ensure column order and types
            for col in ["precision", "recall", "f1-score", "support"]:
                if col not in cr_df.columns:
                    cr_df[col] = np.nan
            cr_df = cr_df[["precision", "recall", "f1-score", "support"]]
            return acc, cm, cr_df

        log_acc, log_cm, log_cr = evaluate_model(y_test, log_reg_pred)
        xgb_acc, xgb_cm, xgb_cr = evaluate_model(y_test, xgb_pred)

        st.subheader("🧪 Model Evaluation")
        st.markdown("**Logistic Regression**")
        st.metric(label="Accuracy (LogReg)", value=f"{log_acc:.2f}")
        st.dataframe(log_cr.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

        st.markdown("**XGBoost**")
        st.metric(label="Accuracy (XGBoost)", value=f"{xgb_acc:.2f}")
        st.dataframe(xgb_cr.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}))

        # Confusion matrices as heatmaps
        cm_labels = ["Class 0", "Class 1"]
        fig_cm1, ax_cm1 = plt.subplots()
        sns.heatmap(log_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm1,
                    xticklabels=cm_labels, yticklabels=cm_labels)
        ax_cm1.set_title('Logistic Regression - Confusion Matrix')
        ax_cm1.set_xlabel('Predicted')
        ax_cm1.set_ylabel('True')
        st.pyplot(fig_cm1)

        fig_cm2, ax_cm2 = plt.subplots()
        sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax_cm2,
                    xticklabels=cm_labels, yticklabels=cm_labels)
        ax_cm2.set_title('XGBoost - Confusion Matrix')
        ax_cm2.set_xlabel('Predicted')
        ax_cm2.set_ylabel('True')
        st.pyplot(fig_cm2)

    except Exception as e:
        st.error(f"Error in image preprocessing or model training: {e}")

    # Utility: Print concise samples to CLI
    def _print_samples(label, items, max_samples=3):
        try:
            print(f"[{label}] collected {len(items)} items.")
            for idx, text in enumerate(items[:max_samples]):
                snippet = (text or "").replace("\n", " ").strip()
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                print(f"  {idx+1}. {snippet}")
        except Exception:
            pass

    # -------------------- 2. Twitter Data Collection --------------------
    tweets = []
    if bearer_token:
        try:
            client = tweepy.Client(bearer_token=bearer_token)
            results = client.search_recent_tweets(query=tweet_query, max_results=tweet_count, tweet_fields=["text"])
            tweets = [tweet.text for tweet in results.data] if results.data else []
            st.success(f"✅ Collected {len(tweets)} tweets.")
            if print_to_cli:
                _print_samples("Twitter", tweets)
        except Exception as e:
            st.error(f"Error fetching tweets: {e}")
    else:
        st.warning("⚠️ Please provide a valid Twitter Bearer Token.")

    # -------------------- 3. News Data Collection --------------------
    news_articles = []
    if news_api_key:
        try:
            news_query = quote_plus(tweet_query if tweet_query else 'pneumonia')
            news_url = f'https://newsapi.org/v2/everything?q={news_query}&apiKey={news_api_key}'
            response = requests.get(news_url, timeout=20)
            news_data = response.json()
            articles = news_data.get('articles', [])
            news_articles = [
                (a.get('title') or '') + " " + (a.get('description') or '')
                for a in articles if a.get('title') or a.get('description')
            ]
            st.success(f"✅ Collected {len(news_articles)} news articles.")
            if print_to_cli:
                _print_samples("News", news_articles)
        except Exception as e:
            st.error(f"Error fetching news: {e}")
    else:
        st.warning("⚠️ Please provide a valid News API Key.")

    # -------------------- 3b. Reddit Data Collection --------------------
    reddit_posts = []
    try:
        reddit_url = f"https://www.reddit.com/search.json?q={quote_plus(tweet_query)}&limit={reddit_count}&sort=new"
        headers = {"User-Agent": "Mozilla/5.0 (StreamlitApp)"}
        r = requests.get(reddit_url, headers=headers, timeout=15)
        if r.status_code == 200:
            rj = r.json()
            children = rj.get("data", {}).get("children", [])
            reddit_posts = [
                (c.get("data", {}).get("title", "") or "") + " " + (c.get("data", {}).get("selftext", "") or "")
                for c in children
            ]
            st.success(f"✅ Collected {len(reddit_posts)} Reddit posts.")
            if print_to_cli:
                _print_samples("Reddit", reddit_posts)
        else:
            st.warning(f"⚠️ Reddit search returned status {r.status_code}.")
    except Exception as e:
        st.error(f"Error fetching Reddit data: {e}")

    # -------------------- 3c. Tavily Web Search --------------------
    tavily_texts = []
    if tavily_api_key:
        try:
            tavily_payload = {
                "api_key": tavily_api_key,
                "query": tweet_query,
                "search_depth": "basic",
                "max_results": tavily_count,
                "include_raw_content": True,
            }
            tavily_resp = requests.post("https://api.tavily.com/search", json=tavily_payload, timeout=20)
            if tavily_resp.status_code == 200:
                tavily_json = tavily_resp.json()
                results = tavily_json.get("results", [])
                tavily_texts = [
                    (res.get("content") or res.get("raw_content") or "") for res in results
                ]
                st.success(f"✅ Collected {len(tavily_texts)} web results (Tavily).")
                if print_to_cli:
                    _print_samples("Web (Tavily)", tavily_texts)
            else:
                st.warning(f"⚠️ Tavily search returned status {tavily_resp.status_code}.")
        except Exception as e:
            st.error(f"Error fetching Tavily results: {e}")
    else:
        st.info("ℹ️ Provide a Tavily API key in `TAVILY_API_KEY` to enable web results.")

    # -------------------- 3d. Wikipedia Search (free) --------------------
    wiki_texts = []
    try:
        wiki_url = (
            f"https://en.wikipedia.org/w/rest.php/v1/search/page?q={quote_plus(tweet_query)}&limit={wiki_count}"
        )
        wresp = requests.get(wiki_url, timeout=20)
        if wresp.status_code == 200:
            wjson = wresp.json()
            pages = wjson.get("pages", [])
            for p in pages:
                title = p.get("title") or ""
                excerpt = p.get("excerpt") or ""
                # Strip HTML tags in excerpt
                excerpt_clean = re.sub(r"<[^>]+>", " ", excerpt)
                wiki_texts.append(f"{title} {excerpt_clean}")
            st.success(f"✅ Collected {len(wiki_texts)} Wikipedia results.")
            if print_to_cli:
                _print_samples("Wikipedia", wiki_texts)
        else:
            st.warning(f"⚠️ Wikipedia search returned status {wresp.status_code}.")
    except Exception as e:
        st.error(f"Error fetching Wikipedia results: {e}")

    # -------------------- 3e. Hacker News Search (free via Algolia API) --------------------
    hn_texts = []
    try:
        hn_url = (
            f"https://hn.algolia.com/api/v1/search?query={quote_plus(tweet_query)}&tags=story&hitsPerPage={hn_count}"
        )
        hresp = requests.get(hn_url, timeout=20)
        if hresp.status_code == 200:
            hjson = hresp.json()
            hits = hjson.get("hits", [])
            for h in hits:
                title = h.get("title") or ""
                story_text = h.get("story_text") or h.get("_highlightResult", {}).get("title", {}).get("value", "") or ""
                story_text_clean = re.sub(r"<[^>]+>", " ", str(story_text))
                hn_texts.append(f"{title} {story_text_clean}")
            st.success(f"✅ Collected {len(hn_texts)} Hacker News stories.")
            if print_to_cli:
                _print_samples("Hacker News", hn_texts)
        else:
            st.warning(f"⚠️ Hacker News search returned status {hresp.status_code}.")
    except Exception as e:
        st.error(f"Error fetching Hacker News results: {e}")

    # -------------------- 3f. PubMed (free E-utilities) --------------------
    pubmed_texts = []
    try:
        esearch = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={quote_plus(tweet_query)}&retmax={pubmed_count}&retmode=json"
        )
        presp = requests.get(esearch, timeout=20)
        if presp.status_code == 200:
            pjson = presp.json()
            ids = pjson.get("esearchresult", {}).get("idlist", [])
            if ids:
                esummary = (
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id=" + ",".join(ids)
                )
                sresp = requests.get(esummary, timeout=20)
                if sresp.status_code == 200:
                    sjson = sresp.json()
                    result = sjson.get("result", {})
                    for pid in ids:
                        rec = result.get(pid, {})
                        title = rec.get("title") or ""
                        pubmed_texts.append(title)
            st.success(f"Collected {len(pubmed_texts)} PubMed records.")
            if print_to_cli:
                _print_samples("PubMed", pubmed_texts)
        else:
            st.warning(f"PubMed search returned status {presp.status_code}.")
    except Exception as e:
        st.error(f"Error fetching PubMed results: {e}")

    # -------------------- 3g. CrossRef (free) --------------------
    crossref_texts = []
    try:
        cr_url = f"https://api.crossref.org/works?query={quote_plus(tweet_query)}&rows={crossref_count}"
        cresp = requests.get(cr_url, timeout=20)
        if cresp.status_code == 200:
            cjson = cresp.json()
            items = cjson.get("message", {}).get("items", [])
            for it in items:
                title_list = it.get("title") or []
                title = title_list[0] if title_list else ""
                abstract = it.get("abstract") or ""
                abstract_clean = re.sub(r"<[^>]+>", " ", abstract)
                crossref_texts.append(f"{title} {abstract_clean}")
            st.success(f"✅ Collected {len(crossref_texts)} CrossRef items.")
            if print_to_cli:
                _print_samples("CrossRef", crossref_texts)
        else:
            st.warning(f"⚠️ CrossRef search returned status {cresp.status_code}.")
    except Exception as e:
        st.error(f"Error fetching CrossRef results: {e}")

    # -------------------- 4. Misinformation Detection --------------------
    def detect_misinformation(texts):
        return [1 if TextBlob(t).sentiment.polarity < 0 else 0 for t in texts]

    tweet_misinfo = detect_misinformation(tweets)
    news_misinfo = detect_misinformation(news_articles)
    reddit_misinfo = detect_misinformation(reddit_posts)
    tavily_misinfo = detect_misinformation(tavily_texts)
    wiki_misinfo = detect_misinformation(wiki_texts)
    hn_misinfo = detect_misinformation(hn_texts)
    pubmed_misinfo = detect_misinformation(pubmed_texts)
    crossref_misinfo = detect_misinformation(crossref_texts)

    total_flags = (sum(tweet_misinfo) + sum(news_misinfo) + sum(reddit_misinfo) + sum(tavily_misinfo)
                   + sum(wiki_misinfo) + sum(hn_misinfo) + sum(pubmed_misinfo) + sum(crossref_misinfo))
    total_count = (len(tweet_misinfo) + len(news_misinfo) + len(reddit_misinfo) + len(tavily_misinfo)
                   + len(wiki_misinfo) + len(hn_misinfo) + len(pubmed_misinfo) + len(crossref_misinfo))
    avg_misinfo_score = (total_flags / (total_count + 1e-5)) if total_count else 0.0
    st.metric("💬 Overall Misinformation Rate", f"{avg_misinfo_score:.2f}")

    # Per-source metrics
    def rate(arr):
        return (sum(arr) / (len(arr) + 1e-5)) if len(arr) else 0.0
    source_rates = {
        "Tweets": rate(tweet_misinfo),
        "News": rate(news_misinfo),
        "Reddit": rate(reddit_misinfo),
        "Web (Tavily)": rate(tavily_misinfo),
        "Wikipedia": rate(wiki_misinfo),
        "Hacker News": rate(hn_misinfo),
        "PubMed": rate(pubmed_misinfo),
        "CrossRef": rate(crossref_misinfo),
    }

    # Sentiment score distributions per source
    def sentiment_scores(texts):
        return [TextBlob(t).sentiment.polarity for t in texts] if texts else []
    tweet_scores = sentiment_scores(tweets)
    news_scores = sentiment_scores(news_articles)
    reddit_scores = sentiment_scores(reddit_posts)
    tavily_scores = sentiment_scores(tavily_texts)
    wiki_scores = sentiment_scores(wiki_texts)
    hn_scores = sentiment_scores(hn_texts)
    pubmed_scores = sentiment_scores(pubmed_texts)
    crossref_scores = sentiment_scores(crossref_texts)

    # -------------------- 5. Agent-Based Model --------------------
    class Patient(Agent):
        def __init__(self, uid, model, misinfo_score=0.5):
            super().__init__(uid, model)
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
            super().__init__(uid, model)
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
    df_reset = df.reset_index()
    if 'Step' in df_reset.columns:
        step_col = 'Step'
    else:
        # Mesa commonly names the second level 'Step' when reset_index is called
        # If not present (edge cases), try to infer column 1 as step
        step_col = df_reset.columns[1] if len(df_reset.columns) > 2 else df_reset.columns[-1]
    sim_means = df_reset.groupby(step_col).agg({
        'Symptom Severity': 'mean',
        'Care Seeking Behavior': 'mean'
    }).reset_index()

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

    # -------------------- 8. New Visualization: Misinformation by Source --------------------
    st.subheader("📊 Misinformation by Source")
    fig3, ax3 = plt.subplots()
    labels = list(source_rates.keys())
    values = [source_rates[k] for k in labels]
    sns.barplot(x=labels, y=values, ax=ax3, palette="viridis")
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Misinformation Rate (0-1)')
    ax3.set_title('Estimated Misinformation Rate by Data Source')
    for idx, val in enumerate(values):
        ax3.text(idx, min(val + 0.02, 0.98), f"{val:.2f}", ha='center', va='bottom', fontsize=9)
    st.pyplot(fig3)

     # -------------------- 9. Sentiment Distributions --------------------
    st.subheader("📈 Sentiment Distributions by Source")
    fig4, ax4 = plt.subplots()
    plotted_any = False
    if tweet_scores:
        sns.kdeplot(tweet_scores, fill=True, label='Tweets', ax=ax4)
        plotted_any = True
    if news_scores:
        sns.kdeplot(news_scores, fill=True, label='News', ax=ax4)
        plotted_any = True
    if reddit_scores:
        sns.kdeplot(reddit_scores, fill=True, label='Reddit', ax=ax4)
        plotted_any = True
    if tavily_scores:
        sns.kdeplot(tavily_scores, fill=True, label='Web (Tavily)', ax=ax4)
        plotted_any = True
    if wiki_scores:
        sns.kdeplot(wiki_scores, fill=True, label='Wikipedia', ax=ax4)
        plotted_any = True
    if hn_scores:
        sns.kdeplot(hn_scores, fill=True, label='Hacker News', ax=ax4)
        plotted_any = True
    if pubmed_scores:
        sns.kdeplot(pubmed_scores, fill=True, label='PubMed', ax=ax4)
        plotted_any = True
    if crossref_scores:
        sns.kdeplot(crossref_scores, fill=True, label='CrossRef', ax=ax4)
        plotted_any = True
    if plotted_any:
        ax4.axvline(0.0, color='k', linestyle='--', alpha=0.6)
        ax4.set_xlim(-1, 1)
        ax4.set_xlabel('TextBlob Sentiment Polarity (-1 to 1)')
        ax4.set_title('Sentiment KDE by Source')
        ax4.legend()
        st.pyplot(fig4)
    else:
        st.info("Not enough text data to plot sentiment distributions.")

    # -------------------- 10. Simulation Trends Over Time --------------------
    st.subheader("⏱️ Simulation Averages Over Time")
    fig5, ax5 = plt.subplots()
    ax5.plot(sim_means[step_col], sim_means['Symptom Severity'], label='Avg Symptom Severity')
    ax5.plot(sim_means[step_col], sim_means['Care Seeking Behavior'], label='Avg Care Seeking Behavior')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Average Value')
    ax5.set_title('Simulation Means by Step')
    ax5.legend()
    st.pyplot(fig5)

    # -------------------- 11. Care-Seeking by Severity (Violin) --------------------
    st.subheader("🎻 Care-Seeking by Severity")
    fig6, ax6 = plt.subplots()
    # Ensure categorical x for better labels
    df_reset['Severity Label'] = df_reset['Symptom Severity'].map({0: 'Mild/None (0)', 1: 'Severe (1)'})
    sns.violinplot(x='Severity Label', y='Care Seeking Behavior', data=df_reset, ax=ax6, inner='box', palette='Set2')
    ax6.set_xlabel('Symptom Severity')
    ax6.set_ylabel('Care Seeking Behavior')
    ax6.set_title('Distribution of Care-Seeking by Severity')
    st.pyplot(fig6)
