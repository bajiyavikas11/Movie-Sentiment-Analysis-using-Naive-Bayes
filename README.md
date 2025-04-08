
# 🎯 Sentiment Analysis of IMDb Reviews using Naive Bayes

In this project I have done sentiment classification on IMDb movie reviews using **Multinomial Naive Bayes**. It includes full preprocessing, vectorization (TF-IDF), training, evaluation, and explainability.

## 📂 Dataset
- **Source**: `tensorflow_datasets.imdb_reviews`
- 25,000 labeled reviews for training, 25,000 for testing

## 🛠️ Tech Stack
- Python
- scikit-learn
- TensorFlow Datasets
- NLTK
- Matplotlib + Seaborn

## 📈 Pipeline Overview

1. **Load Data** from TensorFlow Datasets
2. **Clean & Preprocess** text (remove HTML, punctuation, stopwords)
3. **Vectorize** text using TF-IDF (1-2 grams)
4. **Train** Multinomial Naive Bayes classifier
5. **Evaluate** using accuracy, classification report, confusion matrix
6. **Explain** top contributing words per class
7. **Save** model & vectorizer for future inference

## 📊 Sample Output

```
Accuracy: 0.88
              precision    recall  f1-score   support
    Negative       0.88      0.87      0.87     12500
    Positive       0.88      0.89      0.88     12500

Accuracy: 0.8504
              precision    recall  f1-score   support

    Negative       0.85      0.85      0.85     12500
    Positive       0.85      0.85      0.85     12500

Confusion Matrix:

[[10642  1858]
 [ 1882 10618]]
```

## 📌 To Run in Colab

1. Upload `sentiment_nb.ipynb`
2. Run all cells — no need to download any data manually
3. Model and vectorizer saved as `.pkl` files

## 🧠 Top Words per Sentiment

- **Positive**: great, love, best, excellent, highly
- **Negative**: bad, boring, worst, terrible, waste

## 🏁 Future Work
- Try Logistic Regression or SVM
- Use word embeddings (Word2Vec / BERT)
- Add a web app for real-time review classification

---

📌 Feel free to fork the repo and experiment with the model!
