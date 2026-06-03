import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import ComplementNB

st.set_page_config(page_title="Spam Detector", page_icon="🚫", layout="wide")
st.title("🚫 Spam Detector — Complement Naive Bayes")
st.caption("Built with ComplementNB + TF-IDF (unigrams + bigrams)")

CSV_PATH = '/Users/aanvisawhney/Desktop/vs code/AIML/project1-spam detector/spam.csv'

# ── Load & train (cached so it only runs once) ───────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv(CSV_PATH, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df.drop_duplicates(inplace=True)
    df['length'] = df['message'].apply(len)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    X = df['message']
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    model = ComplementNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=['Ham', 'Spam'],
                                   output_dict=True)
    acc    = accuracy_score(y_test, y_pred)

    return df, vectorizer, model, cm, report, acc

df, vectorizer, model, cm, report, acc = load_and_train()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Model Evaluation", "🔍 Try It"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset at a glance")

    spam_n = df[df['label'] == 'spam'].shape[0]
    ham_n  = df[df['label'] == 'ham'].shape[0]
    total  = len(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total messages",  total)
    c2.metric("Ham (legit)",     ham_n)
    c3.metric("Spam",            spam_n)
    c4.metric("Spam %",          f"{spam_n/total*100:.1f}%")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Message length: spam vs ham**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df[df['label']=='ham']['length'],  bins=30,
                alpha=0.6, color='steelblue', label='Ham')
        ax.hist(df[df['label']=='spam']['length'], bins=30,
                alpha=0.6, color='tomato',    label='Spam')
        ax.set_xlabel('Message Length')
        ax.set_ylabel('Count')
        ax.set_title('Message Length: Spam vs Ham')
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

        avg = df.groupby('label')['length'].mean().round(1)
        st.caption(f"Avg ham length: **{avg['ham']} chars** | Avg spam length: **{avg['spam']} chars**")

    with col_b:
        st.markdown("**Label distribution**")
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        counts = df['label'].value_counts()
        bars = ax2.bar(counts.index, counts.values,
                       color=['steelblue', 'tomato'],
                       edgecolor='white', width=0.5)
        ax2.set_ylabel('Count')
        ax2.set_title('Ham vs Spam count')
        for bar, v in zip(bars, counts.values):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 15,
                     str(v), ha='center', fontweight='bold')
        fig2.tight_layout()
        st.pyplot(fig2)

    st.divider()
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("🟢 **Ham samples**")
        st.dataframe(df[df['label']=='ham'][['message']].head(5),
                     use_container_width=True, hide_index=True)
    with col_s2:
        st.markdown("🔴 **Spam samples**")
        st.dataframe(df[df['label']=='spam'][['message']].head(5),
                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Complement Naive Bayes — evaluation")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",       f"{acc*100:.2f}%")
    m2.metric("Spam Precision", f"{report['Spam']['precision']*100:.1f}%")
    m3.metric("Spam Recall",    f"{report['Spam']['recall']*100:.1f}%")
    m4.metric("Spam F1",        f"{report['Spam']['f1-score']*100:.1f}%")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Confusion matrix**")
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'],
                    linewidths=0.5, cbar_kws={'shrink': 0.8})
        ax3.set_xlabel('Predicted', fontsize=12)
        ax3.set_ylabel('Actual',    fontsize=12)
        ax3.set_title('Confusion Matrix')
        fig3.tight_layout()
        st.pyplot(fig3)

        st.caption(
            f"✅ Spam correctly caught: **{cm[1][1]}**  "
            f"| ❌ Spam missed: **{cm[1][0]}**  "
            f"| ⚠️ Ham wrongly flagged: **{cm[0][1]}**"
        )

    with col2:
        st.markdown("**Classification report**")
        report_df = pd.DataFrame({
            'Class':     ['Ham', 'Spam'],
            'Precision': [f"{report['Ham']['precision']:.3f}",
                          f"{report['Spam']['precision']:.3f}"],
            'Recall':    [f"{report['Ham']['recall']:.3f}",
                          f"{report['Spam']['recall']:.3f}"],
            'F1-Score':  [f"{report['Ham']['f1-score']:.3f}",
                          f"{report['Spam']['f1-score']:.3f}"],
            'Support':   [int(report['Ham']['support']),
                          int(report['Spam']['support'])],
        })
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**What these numbers mean**")
        st.markdown("""
- **Precision** — of all messages flagged spam, how many actually were?
- **Recall** — of all real spam, how many did we catch?
- **F1-Score** — balance between precision and recall
- **False positives** (ham → spam) are worse in real life — legit emails go missing
        """)

    st.divider()
    st.markdown("**Top 20 words most associated with spam**")
    feature_names = vectorizer.get_feature_names_out()
    spam_log_prob  = model.feature_log_prob_[1]
    top_idx        = spam_log_prob.argsort()[-20:][::-1]
    top_words_df   = pd.DataFrame({
        'Word': feature_names[top_idx],
        'Log Probability': spam_log_prob[top_idx].round(3)
    })
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.barh(top_words_df['Word'][::-1],
             top_words_df['Log Probability'][::-1],
             color='tomato', edgecolor='white')
    ax4.set_xlabel('Log Probability')
    ax4.set_title('Top spam-associated words (unigrams + bigrams)')
    fig4.tight_layout()
    st.pyplot(fig4)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Try it yourself")

    examples = [
        "URGENT! You have won a 1 week FREE membership. Call now!",
        "Congratulations! You've won a free iPhone. Click here now!",
        "Hey, are we still on for dinner tonight?",
        "WIN £1000 cash! Text WIN to 87121 now. T&C apply.",
        "Can you send me the notes from today's class?",
    ]

    selected = st.selectbox("Pick an example or write your own:",
                            ["— write your own —"] + examples)
    msg = st.text_area("Message:",
                       value="" if selected == "— write your own —" else selected,
                       height=120)

    if st.button("Check message", type="primary"):
        if msg.strip() == "":
            st.warning("Enter a message first.")
        else:
            tfidf = vectorizer.transform([msg])
            pred  = model.predict(tfidf)[0]
            proba = model.predict_proba(tfidf)[0]
            conf  = round(max(proba) * 100, 1)

            if pred == 1:
                st.error(f"🚨 **SPAM** — {conf}% confident")
            else:
                st.success(f"✅ **Not spam (Ham)** — {conf}% confident")

            st.divider()
            col_p, col_i = st.columns(2)

            with col_p:
                st.markdown("**Probability breakdown**")
                fig5, ax5 = plt.subplots(figsize=(4, 2))
                ax5.barh(['Ham', 'Spam'],
                         [proba[0]*100, proba[1]*100],
                         color=['steelblue', 'tomato'])
                ax5.set_xlim(0, 100)
                ax5.set_xlabel('Probability (%)')
                fig5.tight_layout()
                st.pyplot(fig5)

            with col_i:
                st.markdown("**Raw probabilities**")
                st.dataframe(pd.DataFrame({
                    'Class': ['Ham', 'Spam'],
                    'Probability': [f"{proba[0]*100:.2f}%",
                                    f"{proba[1]*100:.2f}%"]
                }), use_container_width=True, hide_index=True)
