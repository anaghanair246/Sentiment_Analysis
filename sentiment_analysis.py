# Sentiment Analysis using Python (Machine Learning)
# Author: Anagha
# Description: Classifies text as Positive, Negative, or Neutral

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# 1. Sample Dataset 
data = {
    'text': [
        "I love this product, it's amazing!",
        "Worst experience ever, totally disappointed",
        "Not bad, but could be better",
        "Absolutely fantastic! Highly recommend it",
        "Terrible service, I will never come back",
        "It's okay, nothing special"
    ],
    'sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Negative', 'Neutral']
}

df = pd.DataFrame(data)

# 2. Split Data
X = df['text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Text Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Train Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 5. Predict & Evaluate
y_pred = model.predict(X_test_tfidf)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. Test on New Examples
new_texts = [
    "The product quality is great and I love it",
    "This is the worst thing I ever bought",
    "It's fine, I guess"
]

new_tfidf = vectorizer.transform(new_texts)
predictions = model.predict(new_tfidf)

print("\nNew Predictions:")
for text, sentiment in zip(new_texts, predictions):
    print(f"{text} --> {sentiment}")
