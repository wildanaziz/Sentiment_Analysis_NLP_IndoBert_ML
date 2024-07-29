# Sentiment Analysis with IndoBERT and Machine Learning

## Introduction
This project is part of a scholarship submission. The goal is to perform sentiment analysis using IndoBERT and various machine learning algorithms to achieve maximum accuracy. The dataset used in this project was scraped from the Google Play Store.

## Project Structure
The repository is organized as follows:

- `data/`: Contains the scraped dataset.
- `notebooks/`: Jupyter notebooks used for data exploration, preprocessing, and modeling.
- `src/`: Source code for data preprocessing, feature extraction, and model training.
- `models/`: Saved models and results.
- `README.md`: Project documentation.

## Dataset
The dataset consists of reviews scraped from the Google Play Store. It includes reviews in Indonesian, along with their corresponding sentiment labels (positive, negative, neutral).

## Methods
### Data Preprocessing
1. **Text Cleaning**: Removal of HTML tags, special characters, and other noise.
2. **Tokenization**: Splitting text into words or subwords.
3. **Stopwords Removal**: Removal of common words that do not contribute much to the sentiment.
4. **Case Folding**: to lowercase.
5. **toSentence**: merge bunchs of words to sentence.

### Feature Extraction
1. **TF-IDF**: Term Frequency-Inverse Document Frequency used to convert text to numerical features.
2. **Word Embeddings**: Using IndoBERT to obtain contextual word embeddings.

### Modeling
1. **IndoBERT**: Fine-tuning IndoBERT for sentiment classification.
2. **Machine Learning Algorithms**: Training models like Random Forest, SVM, and Logistic Regression on the extracted features.

## Results
The results of the project are as follows:

- **IndoBERT**: Achieved an accuracy of XX%
- **Random Forest**: Achieved an accuracy of XX%
- **SVM**: Achieved an accuracy of XX%
- **Logistic Regression**: Achieved an accuracy of XX%

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sentiment-analysis-indobert-ml.git
    cd sentiment-analysis-indobert-ml
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset (if not included in the repository) and place it in the `data/` directory.

## Usage
To train the models and evaluate their performance, run the notebooks in the `notebooks/` directory in the following order:

1. `Scraping_Dataset.ipynb`
2. `Sentiment-NLP-IndoBert.ipynb.ipynb`
