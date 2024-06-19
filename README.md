# Text Preprocessing and Keyword Extraction

This project involves preprocessing text data and extracting keywords using the TF-IDF (Term Frequency-Inverse Document Frequency) method. The project demonstrates various text preprocessing steps, followed by keyword extraction from a dataset of research papers.

## Features

- **Text Preprocessing**: 
  - Converts text to lower case
  - Removes HTML tags
  - Removes special characters and digits
  - Tokenizes text
  - Removes stopwords
  - Removes words with less than three letters
  - Lemmatizes text
  
- **Keyword Extraction**: 
  - Uses TF-IDF to extract keywords from text
  - Applies CountVectorizer and TfidfTransformer from Scikit-learn
  - Provides functions to sort and extract top keywords from TF-IDF vectors

## Requirements

- Python 3.7+
- NumPy
- Pandas
- NLTK
- Scikit-learn

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/text-preprocessing-keyword-extraction.git
    cd text-preprocessing-keyword-extraction
    ```

2. Install the required packages:
    ```bash
    pip install numpy pandas nltk scikit-learn
    ```

3. Download NLTK data:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

## Usage

1. Load and preprocess the data:
    ```python
    import pandas as pd
    df = pd.read_csv('papers.csv')
    df = df.iloc[:5000,:]
    ```

2. Preprocess text data:
    ```python
    docs = df['paper_text'].apply(lambda x: preprocess_text(x))
    ```

3. Extract keywords using TF-IDF:
    ```python
    idx = 0  # Index of the document for which to extract keywords
    keywords = get_keywords(idx, docs)
    print_results(idx, keywords, df)
    ```

4. Extract keywords from custom text:
    ```python
    custom_text = "Your custom text here..."
    preprocessed_text = preprocess_text(custom_text)
    keywords = get_keywords_from_text(preprocessed_text, docs)
    print("Keywords:", keywords)
    ```

## Functions

- **`preprocess_text(txt)`**: Preprocesses the given text using the steps mentioned above.
- **`sort_coo(coo_matrix)`**: Sorts the TF-IDF vectors by descending order of scores.
- **`extract_topn_from_vector(feature_names, sorted_items, topn=10)`**: Extracts top N keywords from the sorted TF-IDF vectors.
- **`get_keywords(idx, docs)`**: Generates TF-IDF for the given document and extracts top keywords.
- **`get_keywords_from_text(preprocessed_text, docs)`**: Generates TF-IDF for the given preprocessed text and extracts top keywords.
- **`print_results(idx, keywords, df)`**: Prints the title, abstract, and keywords of the document at the given index.

## Example

Extracting keywords from a document in the dataset:
```python
idx = 590
keywords = get_keywords(idx, docs)
print_results(idx, keywords, df)
