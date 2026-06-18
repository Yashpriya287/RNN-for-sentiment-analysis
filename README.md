# Sentiment Analysis using RNN on IMDB Movie Reviews

## Overview

This project implements a Recurrent Neural Network (RNN) using PyTorch to perform sentiment analysis on movie reviews from the IMDB dataset. The objective is to classify reviews as either positive or negative based on their textual content.

The project demonstrates the complete Natural Language Processing (NLP) pipeline, including text preprocessing, feature extraction using TF-IDF, model training, and evaluation.

---

## Dataset

The project uses the IMDB Movie Reviews Dataset.

### Dataset Details

- 50,000 movie reviews
- Binary sentiment classification
- Classes:
  - Positive
  - Negative

---

## Data Preprocessing

The following preprocessing techniques were applied to improve text quality:

- Lowercase conversion
- HTML tag removal
- URL removal
- Punctuation removal
- Stopword removal using NLTK
- Stemming using Porter Stemmer
- Label encoding for sentiment classes

These steps help reduce noise and improve model performance.

---

## Feature Extraction

### TF-IDF Vectorization

Text data was transformed into numerical representations using TF-IDF (Term Frequency-Inverse Document Frequency).

```python
TfidfVectorizer(max_features=5000)
```

This converts textual reviews into feature vectors that capture the importance of words within the dataset.

---

## Model Architecture

### Recurrent Neural Network (RNN)

Architecture:

- Input Layer
- Simple RNN Layer
- Fully Connected Layer
- Sigmoid Activation Layer

Workflow:

```text
Movie Review
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
RNN Layer
      ↓
Fully Connected Layer
      ↓
Sentiment Prediction
```

---

## Training Configuration

| Parameter | Value |
|------------|--------|
| Hidden Size | 128 |
| Number of Layers | 1 |
| Batch Size | 64 |
| Optimizer | Adam |
| Loss Function | Binary Cross Entropy Loss |
| Epochs | 10 |

---

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy
- NLTK
- Scikit-Learn
- Jupyter Notebook

---

## Key Concepts Covered

- Natural Language Processing (NLP)
- Text Cleaning and Preprocessing
- TF-IDF Vectorization
- Recurrent Neural Networks (RNN)
- Binary Classification
- Deep Learning with PyTorch
- Sentiment Analysis

---

## Results

The RNN model successfully learns sentiment patterns from movie reviews and classifies unseen reviews into positive and negative categories.

Evaluation Metric:

- Accuracy
85.53%
---

## Project Workflow

```text
IMDB Dataset
      ↓
Data Cleaning
      ↓
Token Processing
      ↓
TF-IDF Feature Extraction
      ↓
Train-Test Split
      ↓
RNN Training
      ↓
Model Evaluation
      ↓
Sentiment Prediction
```

---

## Future Improvements

- Implement LSTM Networks
- Implement GRU Networks
- Use Word Embeddings (Word2Vec, GloVe)
- Bidirectional RNN/LSTM
- Transformer-based Architectures
- Hyperparameter Optimization

---

## Conclusion

This project demonstrates the application of Recurrent Neural Networks for sentiment analysis on textual data. Through text preprocessing, TF-IDF feature extraction, and deep learning-based classification, the model effectively predicts the sentiment of movie reviews and provides a strong foundation for more advanced NLP systems.

---

## Author

**Yashpriya Dwivedi**
