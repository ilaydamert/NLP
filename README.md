# NLP - BoW, TF-IDF, and N-gram

This repository demonstrates the use of three common text vectorization techniques in Natural Language Processing (NLP): **Bag of Words (BoW)**, **Term Frequency-Inverse Document Frequency (TF-IDF)**, and **N-grams**. These techniques are commonly used for text feature extraction in machine learning models.

## About

In this project, we explore three popular methods for transforming raw text data into numerical representations that can be used by machine learning algorithms:

1. **Bag of Words (BoW):** A simple method that represents text by counting the frequency of words within the text. It does not consider the order of the words.
   
2. **Term Frequency-Inverse Document Frequency (TF-IDF):** A statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). It is used to highlight important words while reducing the weight of common words.

3. **N-grams:** This method considers sequences of `n` consecutive words in a document. For example:
   - **Unigrams (1-grams)**: Single words like "hello", "world".
   - **Bigrams (2-grams)**: Pairs of consecutive words like "hello world", "machine learning".
   - **Trigrams (3-grams)**: Triplets of consecutive words like "I love programming", "deep learning models".
   
   N-grams are useful for capturing context and relationships between words, especially for tasks like sentiment analysis and text classification.

These techniques are fundamental for many text classification tasks, such as spam detection, sentiment analysis, and topic modeling.
