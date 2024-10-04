# Text_Sentiment_Analysis
Stock-Market Sentiment Dataset for Twitter Stock News Classification

# 📊 Text Sentiment Analysis: Stock-Market Sentiment Dataset

## Overview
This project aims to classify stock market sentiments from Twitter comments into two categories: **`negative`** and **`positive`**. We will utilize the Stock-Market Sentiment Dataset available on [Kaggle](http://www.kaggle.com/datasets/yash612/stockmarket-sentiment-dataset) and compare the performance of two models: **Word2Vec** and **BERT**.

## Objectives
- Classify tweets related to stocks as either **`negative`** or **`positive`**.
- Compare the performance of **Word2Vec** and **BERT** models in sentiment classification.

## Implementation Steps

### 1. Set Up Google Colab
- Create a new notebook in Google Colab.
- Ensure access to necessary libraries:
  ```python
  !pip install pandas numpy gensim transformers scikit-learn nltk
  
2. Load the Dataset
  ```python
import pandas as pd
url = "stockmarket_sentiment_dataset.csv"
data = pd.read_csv(url)

3. Data Preprocessing
  ```python
Clean and preprocess the tweet text by removing URLs, special characters, and stop words.

import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|#', '', text)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

data['cleaned_text'] = data['tweet_text'].apply(clean_text)
4. Word2Vec Model
Train a Word2Vec model on the cleaned text.

from gensim.models import Word2Vec

sentences = [tweet.split() for tweet in data['cleaned_text']]
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
5. BERT Model
Use a pre-trained BERT model for sentiment classification.

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Prepare dataset and dataloaders
dataset = SentimentDataset(data['cleaned_text'].tolist(), data['sentiment'].tolist())
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
6. Model Training and Evaluation
Train both models and evaluate their performance using metrics such as accuracy, precision, recall, and F1-score.

from sklearn.metrics import classification_report

# Train and evaluate Word2Vec model
# Train and evaluate BERT model
7. Comparison of Results
Compare the performance of both models based on evaluation metrics.

# Print results for both models
print("Word2Vec Model Results: ")
print(classification_report(y_true, y_pred_w2v))

print("BERT Model Results: ")
print(classification_report(y_true, y_pred_bert))
8. Documentation and GitHub Repository
Document your code and results clearly.
Push the notebook and related files to your GitHub repository.
Conclusion
This project analyzes and classifies sentiments from stock market tweets using two different models. The comparison of Word2Vec and BERT will provide insights into their effectiveness in sentiment analysis tasks.

Future Work
Explore additional models and techniques for improved accuracy.
Implement real-time sentiment analysis using live Twitter data.
Acknowledgments
Special thanks to the contributors of the Stock-Market Sentiment Dataset on Kaggle for providing the data used in this project.


- **Bold Text**: Emphasized important terms and model names.
- **Conclusion and Acknowledgments**: Included sections to summarize the project and give credit.

This structure is ready for your GitHub repository and will enhance both readability and professionalism. Let me know if you need further modifications or additions!
