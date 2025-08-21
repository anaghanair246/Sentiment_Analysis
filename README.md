# Sentiment_Analysis
Classify text as Positive, Negative, or Neutral using Python and Machine Learning
# Sentiment Analysis using Python (Machine Learning)

This project is a simple **Sentiment Analysis tool** that classifies text into **Positive**, **Negative**, or **Neutral** categories using **Machine Learning**. It is built with **Python**, **scikit-learn**, and **TF-IDF vectorization**.

---

## ✅ Features
- Preprocesses text data and converts it into numerical vectors using **TF-IDF**.
- Trains a **Naive Bayes classifier** for sentiment classification.
- Evaluates model performance using a classification report.
- Allows testing on new text inputs for real-time predictions.

---

## ✅ Technologies Used
- **Python 3.x**
- **Pandas** – for handling data
- **scikit-learn** – for machine learning model and TF-IDF
- **NumPy** – for numerical operations

---

## ✅ Dataset
The project uses a **sample dataset of sentences with sentiment labels (Positive, Negative, Neutral)**. You can replace this with a real dataset like:
- [Twitter Sentiment Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [Amazon Reviews Dataset](https://www.kaggle.com/bittlingmayer/amazonreviews)

---

## ✅ How It Works
1. **Data Loading**: Reads text and sentiment labels into a Pandas DataFrame.
2. **Data Splitting**: Divides the data into training and testing sets.
3. **Vectorization**: Converts text into TF-IDF vectors.
4. **Model Training**: Trains a **Multinomial Naive Bayes** model.
5. **Evaluation**: Prints a classification report for accuracy, precision, recall, and F1-score.
6. **Prediction**: Predicts sentiment for new custom text inputs.

---

