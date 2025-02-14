# Sentiment Analysis on Movie Reviews

## Project Overview

This project focuses on developing a deep learning-based Natural Language Processing (NLP) system to perform binary sentiment analysis on publicly available movie reviews. The system is designed to determine whether a review expresses positive or negative sentiment using a Recurrent Neural Network (RNN) with bidirectional LSTM.

## Objectives

The objectives of this project are to:
- Collect movie review data from publicly available datasets.
- Preprocess the data using various NLP techniques.
- Train a binary classification model.
- Evaluate the model's performance.
- Provide an interactive feature for real-time sentiment prediction on user-input movie reviews.

## Methodology

### Data Preparation
- Download and preprocess datasets using the Pandas library.
- Remove duplicates and irrelevant columns, standardize ratings, and clean reviews using NLP techniques.
- Combine datasets and balance positive and negative sentiments.

### Tokenization
- Convert text reviews into numerical representations.
- Prioritize key words, filter non-English words, and remove named entities.

### Model Development
- Use an embedding layer, spatial dropout layer, bidirectional LSTM layer, dropout layer, and dense output layer.

### Model Evaluation
- Evaluate performance using another dataset
- Analyze errors and visualize training history.
- Provide an interactive feature for user-based evaluation.

## Setup and Tools

### Environment
- Developed using Python 3.10 and TensorFlow 2.16 within an Anaconda virtual environment.

### Setup
Follow these steps to set up the environment:

1. **Create the Environment**:
    ```bash
    conda env create -f environment.yml
    ```

2. **Activate the Environment**:
    ```bash
    conda activate Movie_Sentiment_Analysis
    ```

### Libraries
- **NLP Libraries**: NLTK, SpaCy, Regular Expressions (re), Enchant
- **Deep Learning Libraries**: TensorFlow, Keras, Keras Tuner
- **Evaluation Libraries**: Scikit Learn, Seaborn, Matplotlib
- **Supporting Libraries**: Pandas, NumPy, Pickle, JSON, OS

### Dataset
- Combined datasets from Stanford Sentiment Treebank (SST-2), Stanford IMDB, Rotten Tomatoes, Arize AI, and IMDB Rating Dataset.

## Results

### Usage
Once the environment is set up, you can run `Main.py` or `Main.ipynb`

### Challenges
- Managing hardware and computational limitations.
- Handling neutral sentiments.
- Tuning model parameters.

### Comparison
- Optimizers: Nadam optimizer performed best.
- Model Architecture: Bidirectional LSTM outperformed simpler architectures.
- Regularization Methods: Dropout layers were effective in avoiding overfitting.
- Preprocessing Strategies: Optimized to reduce computational overhead.
