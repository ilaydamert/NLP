import pandas as pd
import re
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from nltk.corpus import stopwords
from textblob import TextBlob
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer

df = pd.read_csv("poem_dataset.csv")
documents = df["Poem"]
labels = df["Genre"]

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))

def preprocessing(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = " ".join(text.split())
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = " ".join([word for word in text.split() if len(word) > 2])
    text = str(TextBlob(text).correct())
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w, pos="v") for w in tokens]
    text = [word for word in lemmatized if word not in stop_words]
    return " ".join(text)


processed_documents = [preprocessing(doc) for doc in documents]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_documents)
feature_names = vectorizer.get_feature_names_out()
df_bow = pd.DataFrame(X.toarray(), columns = feature_names)

word_freq = dict(zip(feature_names, X.sum(axis = 0).A1))
most_common_words = Counter(word_freq).most_common(5)
                         


