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

## References
- Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013). Recursive deep models for semantic compositionality over a sentiment treebank. Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, 1631-1642. Association for Computational Linguistics. https://www.aclweb.org/anthology/D13-1170
- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, 142-150. Association for Computational Linguistics. http://www.aclweb.org/anthology/P11-1015
- Pang, B., & Lee, L. (2005). Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. Proceedings of the Association for Computational Linguistics (ACL). https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes
- Arize AI. (n.d.). Movie reviews with context drift [Dataset]. Hugging Face. Retrieved January 29, 2025, from https://huggingface.co/datasets/arize-ai/movie_reviews_with_context_drift
- itsabba3. (n.d.). IMDB rating dataset [Dataset]. Kaggle. Retrieved January 29, 2025, from https://www.kaggle.com/datasets/itsabba3/imdb-rating
