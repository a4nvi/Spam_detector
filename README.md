# 🚫 Spam Detector
http://localhost:8501
A machine learning model that classifies SMS messages as **spam** or **ham (legitimate)** using Complement Naive Bayes and TF-IDF vectorization — built as Project 1 of my AI/ML learning roadmap.

---

## 📌 What it does

- Trains on 5,500+ real SMS messages from the SMS Spam Collection dataset
- Converts raw text into numerical features using TF-IDF (unigrams + bigrams)
- Classifies messages as spam or ham with ~98% accuracy
- Includes a full Streamlit web app to explore the data, evaluate the model, and test live predictions

---

## 🧠 What I learned building this

- How a machine learning model actually "learns" from labeled data
- Why you **never** fit the vectorizer on test data — only transform it
- The difference between **accuracy**, **precision**, **recall**, and **F1-score** — and why accuracy alone is misleading
- What a **confusion matrix** shows: false positives (ham flagged as spam) vs false negatives (spam that got through)
- Why **Complement Naive Bayes** outperforms standard Multinomial NB on imbalanced text datasets
- How **TF-IDF** with bigrams captures context that single words miss (e.g. "free" vs "click free now")
- How **stratified splitting** ensures the train/test split preserves the original spam/ham ratio
- How to save a trained model with `pickle` and load it into a separate app

---

## 🛠 Tech stack

| Tool | Purpose |
|---|---|
| `pandas` | Loading, cleaning, and exploring the dataset |
| `scikit-learn` | TF-IDF vectorization, train/test split, model training, evaluation |
| `ComplementNB` | Classification model — better than MultinomialNB on imbalanced data |
| `matplotlib` + `seaborn` | Data visualization and confusion matrix heatmap |
| `pickle` | Saving and loading the trained model |
| `streamlit` | Interactive web app |

---

## 📊 Model performance

| Metric | Ham | Spam |
|---|---|---|
| Precision | 0.99 | 0.97 |
| Recall | 0.99 | 0.96 |
| F1-Score | 0.99 | 0.97 |
| **Overall Accuracy** | | **~98%** |

- **False positives** (ham wrongly flagged as spam): very low — critical in real email systems
- **False negatives** (spam that slipped through): minimal

---

## 🗂 Project structure

```
project1-spam detector/
│
├── project1.py       # Main ML pipeline: EDA, training, evaluation
├── app.py            # Streamlit web app
├── spam.csv          # Dataset (SMS Spam Collection)
├── spam_model.pkl    # Saved trained model
├── vectorizer.pkl    # Saved TF-IDF vectorizer
└── README.md
```

---

## ▶️ How to run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/spam-detector.git
cd spam-detector
```

**2. Install dependencies**
```bash
pip install pandas scikit-learn matplotlib seaborn streamlit
```

**3. Run the training script first** (generates the `.pkl` files)
```bash
python project1.py
```

**4. Launch the Streamlit app**
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 📱 App features

- **Data Overview** — message length distribution, spam vs ham counts, sample messages
- **Model Evaluation** — confusion matrix heatmap, classification report, top spam-associated words
- **Try It** — paste any message and get an instant spam/ham prediction with confidence score

---

## 📁 Dataset

**SMS Spam Collection Dataset**
- 5,574 SMS messages labeled as spam or ham
- Source: [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) / UCI ML Repository
- Class imbalance: ~87% ham, ~13% spam — handled via stratified splitting and ComplementNB

---

## 🔑 Key decisions explained

**Why ComplementNB over MultinomialNB?**
ComplementNB is specifically designed for imbalanced datasets. Since only 13% of messages are spam, standard NB gets biased toward predicting ham. ComplementNB corrects for this by training on the complement of each class.

**Why bigrams (`ngram_range=(1,2)`)?**
Single words like "free" or "win" appear in both spam and legitimate messages. Bigrams capture context — "win cash", "click now", "free membership" are far stronger spam signals than any single word alone.

**Why stratified splitting?**
Without `stratify=y`, a random split might put most spam in training and leave very little in the test set — making evaluation unreliable. Stratification ensures the 87/13 ratio is preserved in both splits.

---

*Part of my AI/ML learning roadmap — building every concept through projects.*
