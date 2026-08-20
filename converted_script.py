#!/usr/bin/env python
# coding: utf-8

# In[89]:


# 📦 Imports
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns


# In[91]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud


# In[93]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[95]:


import joblib
import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')


# In[97]:


# 🧹 Load & Clean Data
df = pd.read_csv("spam.csv", encoding='ISO-8859-1')[['label', 'text']]
df.columns = ['label', 'text']


# In[99]:


# 🧽 Text Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# In[101]:


def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)


# In[103]:


df['clean_text'] = df['text'].apply(clean_text)
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})


# In[104]:


# 📊 Visualizations
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df)
plt.title("Ham vs Spam")
plt.xlabel("Message Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("eda/ham_vs_spam.png", dpi=300)
plt.show()
plt.close()


# In[107]:


df['text_length'] = df['text'].apply(len)

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='text_length', hue='label', bins=50, kde=True)
plt.title("Message Length by Label")
plt.xlabel("Message Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("eda/message_length.png", dpi=300)
plt.show()
plt.close()


# In[109]:


spam_words = ' '.join(df[df['label'] == 'spam']['clean_text'])
ham_words = ' '.join(df[df['label'] == 'ham']['clean_text'])


# In[111]:
spam_wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(spam_words)

spam_wc.to_file("eda/spam_wordcloud.png")


ham_wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(ham_words)

ham_wc.to_file("eda/ham_wordcloud.png")


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(spam_wc)
plt.title("Spam Word Cloud")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(ham_wc)
plt.title("Ham Word Cloud")
plt.axis("off")

plt.tight_layout()
plt.savefig("eda/wordcloud_comparison.png", dpi=300)
plt.show()
plt.close()






# In[115]:


# 🔀 Data Split
X = df['clean_text']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


# In[117]:


# 📈 TF-IDF
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2))


# In[119]:


# TF-IDF
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)


# In[121]:


# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# In[123]:


# Save model and vectorizer
joblib.dump(model, 'spam_detector_model.pkl')
joblib.dump(tfidf, 'vectorizer.pkl')

print("✅ Model and Vectorizer saved successfully.")


# In[125]:


# 🤖 Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(alpha=0.3),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20),
    "Decision Tree": DecisionTreeClassifier(max_depth=20),
    "SVM": LinearSVC()
}


# In[127]:


results = {}


# In[129]:


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


# In[131]:


# 📊 Results dictionary
results = {}


# In[133]:


# 🔁 Train, Predict and Show Confusion Matrix
for name, model in models.items():
    pipe = Pipeline([('tfidf', tfidf), ('clf', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    # 📢 Print Report
    print(f"\n📌 {name}")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # 📉 Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f"Confusion Matrix - {name}")

    plt.tight_layout()

    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(f"eda/{filename}", dpi=300)

    display(fig)
    plt.close(fig)


# In[135]:


# 📊 Accuracy Comparison
plt.figure(figsize=(8, 5))

sns.barplot(
    x=list(results.keys()),
    y=list(results.values())
)

plt.ylim(0.9, 1.0)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("eda/model_accuracy_comparison.png", dpi=300)
plt.show()
plt.close()


# In[41]:


# 🎯 Best Model: Improved Naive Bayes (Based on Performance + Simplicity)
best_model = Pipeline([('tfidf', tfidf), ('clf', MultinomialNB(alpha=0.3))])
best_model.fit(X_train, y_train)


# In[143]:


# Save model and vectorizer
joblib.dump(model, 'spam_detector_model.pkl')


# In[145]:


joblib.dump(tfidf, 'vectorizer.pkl')


# In[141]:


# 💾 Save Model
joblib.dump(best_model, "naive_bayes_model.pkl")


# In[45]:


# 🔍 Predict Function
def predict_spam(text):
    model = joblib.load("naive_bayes_model.pkl")
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    return "Spam ❌" if prediction else "Ham ✅"


# In[47]:


# 🧪 Example Predictions
print(predict_spam("You’ve won ₹10,000 cash prize! Click here to claim."))
print(predict_spam("Let’s meet tomorrow for the assignment."))







