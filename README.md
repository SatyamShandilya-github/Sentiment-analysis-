# 🎬 SentimentAI — Movie Review Sentiment Analysis

A machine learning based sentiment analysis application that predicts whether a movie review is **Positive** or **Negative** using Natural Language Processing (NLP).

The project uses **TF-IDF feature extraction** with Unigrams and Bigrams and a **Logistic Regression** classifier. A Streamlit-based graphical interface allows users to enter reviews and receive predictions along with model confidence and sentiment probabilities.

---

## 🚀 Project Overview

Sentiment analysis is a Natural Language Processing task used to determine the emotional tone of a piece of text.

In this project, a machine learning model is trained on IMDB movie reviews to classify reviews into two categories:

- 😊 Positive
- 😞 Negative

The complete workflow includes:

**Data → Preprocessing → TF-IDF → Model Training → Evaluation → Model Saving → GUI**

---

## ✨ Features

- 🎬 Movie review sentiment prediction
- 🧠 Machine Learning based classification
- 🔤 TF-IDF with Unigrams + Bigrams
- 📊 Model confidence and sentiment probabilities
- 📈 Review statistics
- 🖥️ Interactive Streamlit GUI
- 💾 Saved trained model and vectorizer
- 📓 Complete Jupyter Notebook containing the ML workflow
- 📊 Comparison of multiple classification algorithms

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and NLP |
| TF-IDF | Text feature extraction |
| Logistic Regression | Final classification model |
| Naive Bayes | Model comparison |
| Linear SVM | Model comparison |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive GUI |
| Jupyter Notebook | Model development and experimentation |

---

## 📂 Project Structure

```text
Sentiment-analysis-/
│
├── data/
│   └── IMDB Dataset.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── sentiment_analysis.ipynb
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
