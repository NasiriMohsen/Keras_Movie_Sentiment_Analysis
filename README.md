# Keras_Movie_Sentiment_Analysis
Movie review sentiment analysis using Keras and Stanford datasets

## Prerequisites
Make sure to install Anaconda before proceeding.

## Setup
Follow these steps to set up the environment:

1. **Create the Environment**:
    ```bash
    conda env create -f environment.yml
    ```

2. **Activate the Environment**:
    ```bash
    conda activate Movie_Sentiment_Analysis
    ```

## Usage
Once the environment is set up, you can run Main.py or Main.ipynb.

# Sentiment Analysis on Movie Reviews

## Project Overview

This project focuses on developing a deep learning-based Natural Language Processing (NLP) system to perform binary sentiment analysis on publicly available movie reviews. The system is designed to determine whether a review expresses positive or negative sentiment using a Recurrent Neural Network (RNN) with bidirectional LSTM.

## Objectives

The objectives of this project are to:
- Collect movie review data from publicly available datasets.
- Preprocess the data using various NLP techniques.
- Train a binary classification model using a bidirectional LSTM network.
- Evaluate the model's performance using accuracy, precision, recall, F1-score, and confusion matrices.
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
- Train the model to capture bidirectional context for sentiment analysis.

### Model Evaluation
- Evaluate performance using metrics like accuracy, precision, recall, and F1-score.
- Analyze errors and visualize training history.
- Provide an interactive feature for user-based evaluation.

## Setup and Tools

### Environment
- Developed using Python 3.10 and TensorFlow 2.16 within an Anaconda virtual environment.

### Libraries
- **NLP Libraries**: NLTK, SpaCy, Regular Expressions (re), Enchant
- **Deep Learning Libraries**: TensorFlow, Keras, Keras Tuner
- **Evaluation Libraries**: Scikit Learn, Seaborn, Matplotlib
- **Supporting Libraries**: Pandas, NumPy, Pickle, JSON, OS

### Dataset
- Combined datasets from Stanford Sentiment Treebank (SST-2), Stanford IMDB, Rotten Tomatoes, Arize AI, and IMDB Rating Dataset.

## Results

### Performance
- High accuracy and performance on both training and validation datasets.
- Detailed evaluation metrics and confusion matrices included.

### Comparison
- Optimizers: Nadam optimizer performed best.
- Model Architecture: Bidirectional LSTM outperformed simpler architectures.
- Regularization Methods: Dropout layers were effective in avoiding overfitting.
- Preprocessing Strategies: Optimized to reduce computational overhead.

## Challenges
- Managing hardware and computational limitations.
- Handling neutral sentiments.
- Tuning model parameters.

## Conclusion
The project successfully developed a sentiment analysis model with a bidirectional LSTM architecture, providing valuable insights into the process and ways to enhance it.

## GitHub Link
You can access the project on GitHub: [Sentiment Analysis on Movie Reviews](https://github.com/NasiriMohsen/Keras_Movie_Sentiment_Analysis)

