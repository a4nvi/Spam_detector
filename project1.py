# RUN IN INTERACTIVE MODE (JUPYTER NOTEBOOK OR PYTHON SHELL)

# BASIC SETUP
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.naive_bayes import ComplementNB

df = pd.read_csv(
    '/Users/aanvisawhney/Desktop/vs code/AIML/project1-spam detector/spam.csv',
    encoding='latin-1'
)
df=df[['v1','v2']]
df.columns = ['label', 'message']
print(df.head(10))
print(df['label'].value_counts())
print(df.isnull().sum())
print("duplicates: ", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# EXPLORATORY DATA ANALYSIS
print("total messages: ", len(df))
spam_count = len(df[df['label'] == 'spam'])
ham_count = len(df[df['label'] == 'ham'])
print("spam messages: ", spam_count)
print("ham messages: ", ham_count)
df['length']= df['message'].apply(len)         #feature engineering: find length of message 
print(df.groupby('label')['length'].mean())           #avg message length
plt.figure(figsize=(10, 5))
plt.hist(df[df['label']=='ham']['length'], bins=30, alpha=0.6, color='steelblue', label='Ham')
plt.hist(df[df['label']=='spam']['length'], bins=30, alpha=0.6, color='tomato', label='Spam')
plt.xlabel('Message Length')
plt.ylabel('Count')
plt.title('Message Length: Spam vs Ham')
plt.legend()
plt.show()

# TEXT TO NUMBERS
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})     #labels to numbers
X = df['message']                                            #features are the messages
y = df['label_num']                                          #Target/output are the numeric labels

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
vectorizer=TfidfVectorizer(
    stop_words='english',               #remove common words that don't add much meaning (is,are etc.)  
    ngram_range=(1, 2)          #consider single words (unigrams) and pairs of words (bigrams) to capture more context

)
X_train_tfidf=vectorizer.fit_transform(X_train)     #fit and transform training data
X_test_tfidf=vectorizer.transform(X_test)           #transform test data using same vectorizer (don't fit again)

#This is a critical rule — never let your model see test data during training.

print("Training samples:", X_train_tfidf.shape)     # (Messages × Words)

# First, I converted the target labels from text to numerical values. 
# Then I split the dataset into training and testing sets. 
# Since machine learning models cannot directly process text, 
# I used TF-IDF vectorization to convert messages into numerical feature vectors. 
# These vectors were then used to train the Naive Bayes classifier.

#TRAIN MODEL

model = ComplementNB()
model.fit(X_train_tfidf, y_train)
y_pred=model.predict(X_test_tfidf)                          #predict() method generates predictions for the test set based on the learned patterns.
print("accuracy: ", accuracy_score(y_test, y_pred))         #accuracy is the ratio of correctly predicted messages to total messages in the test set.
print("classification report:\n", classification_report(y_test,y_pred, target_names=['ham', 'spam']))     #precision, recall, f1-score for each class
print("confusion_matrix:\n", confusion_matrix(y_test,y_pred))

import pickle

pickle.dump(model, open('spam_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
# Visualise confusion matrix

cm= confusion_matrix(y_test,y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['ham', 'spam'], 
            yticklabels=['ham', 'spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('confusion matrix')
plt.show()
print("False positives (ham flagged as spam):", cm[0][1])
print("False negatives (spam that got through):", cm[1][0])

# Self Testing
def predict_spam(message):
    message_tfidf=vectorizer.transform([message])
    pred=model.predict(message_tfidf)
    print(model.predict_proba(message_tfidf))  #probability of being ham or spam    
    return 'spam' if pred[0] == 1 else 'ham'
test_message = "URGENT! You have won a 1 week FREE membership. Call now!"
print("Test message:", test_message)
print("Predicted label:", predict_spam(test_message))
print("predicted label:", predict_spam("Congratulations! You've won a free iPhone. Click here now!"))
print("Predicted label:", predict_spam("Hey, are we still on for dinner tonight?"))